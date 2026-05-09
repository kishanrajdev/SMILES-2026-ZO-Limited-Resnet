"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args:
        model:            The ``nn.Module`` to optimize.
        lr:               Step size / learning rate.
        eps:              Perturbation magnitude for the finite-difference
                          estimator.
        perturbation_mode: Distribution used to sample the perturbation
                          direction. ``"gaussian"`` draws from N(0, I);
                          ``"uniform"`` draws from U(-1, 1) and normalises.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
           Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
           implement dynamic layer selection strategies.

    Example — tune only the final linear layer::

        optimizer = ZeroOrderOptimizer(model)
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        eps: float = 1e-3,
        perturbation_mode: str = "gaussian",
        use_spsa: bool = True,
        momentum: float = 0.9,
        eps_scheduler: str = "const",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.use_spsa = use_spsa
        self.momentum = momentum
        self.eps_scheduler = eps_scheduler
        self.step_count = 0

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        # Momentum accumulator for each parameter
        self.momentum_buffer: dict[str, torch.Tensor] = {}

        # ------------------------------------------------------------------
        # STUDENT: Set self.layer_names to the parameters you want to tune.
        #
        # The default below selects only the final classification head.
        # You may replace this with any subset of named parameters, e.g.:
        #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        #
        # You can also update self.layer_names inside .step() to implement
        # a dynamic schedule (e.g. gradually unfreeze deeper layers).
        # ------------------------------------------------------------------
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Sample a random perturbation vector of the same shape as ``param``.

        Args:
            param: The parameter tensor whose shape determines the output shape.
            normalize: If True, normalize to unit L2 norm. For SPSA, often no normalization
                      is needed since the finite-difference estimator scales it anyway.

        Returns:
            A tensor of the same shape as ``param``.
        """
        if self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        else:  # uniform
            u = torch.rand_like(param) * 2.0 - 1.0

        if normalize:
            norm = u.norm()
            if norm > 0:
                u = u / norm
        return u

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate pseudo-gradients using SPSA or per-parameter estimation.

        SPSA (Simultaneous Perturbation Stochastic Approximation):
            Uses only 2 forward passes regardless of model size by perturbing
            all parameters simultaneously with the same random direction.
            - Sample a single random direction ``u`` for all parameters.
            - Evaluate f_plus and f_minus with all parameters perturbed by ±eps*u.
            - Compute gradient as (f_plus - f_minus) / (2*eps) * u for each param.

        Args:
            loss_fn: Callable that evaluates the objective on the current batch.
            params:  Dict of active parameter name → tensor.

        Returns:
            Dict mapping each parameter name to its estimated pseudo-gradient.
        """
        grads: dict[str, torch.Tensor] = {}

        if self.use_spsa:
            # SPSA: single shared perturbation for all parameters
            with torch.no_grad():
                # Sample a shared random direction for all parameters
                perturbations = {}
                for name, param in params.items():
                    # Don't normalize for SPSA; let the finite-difference scale it
                    perturbations[name] = self._sample_direction(param, normalize=False)

                # f(x + eps * u)
                for name, param in params.items():
                    param.data.add_(self.eps * perturbations[name])
                f_plus = loss_fn()

                # f(x - eps * u)
                for name, param in params.items():
                    param.data.sub_(2.0 * self.eps * perturbations[name])
                f_minus = loss_fn()

                # Restore original values
                for name, param in params.items():
                    param.data.add_(self.eps * perturbations[name])

                # Compute gradient estimate for each parameter
                grad_scale = (f_plus - f_minus) / (2.0 * self.eps)
                for name, param in params.items():
                    grads[name] = grad_scale * perturbations[name]
        else:
            # Fallback: per-parameter central-difference estimator
            with torch.no_grad():
                for name, param in params.items():
                    u = self._sample_direction(param, normalize=True)

                    # f(x + eps * u)
                    param.data.add_(self.eps * u)
                    f_plus = loss_fn()

                    # f(x - eps * u)
                    param.data.sub_(2.0 * self.eps * u)
                    f_minus = loss_fn()

                    # Restore original value
                    param.data.add_(self.eps * u)

                    grad_estimate = ((f_plus - f_minus) / (2.0 * self.eps)) * u
                    grads[name] = grad_estimate

        return grads

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Uses momentum-based descent with optional gradient clipping for stability.
        - If momentum > 0: accumulate exponential moving average of gradients.
        - Optionally clip large gradient updates for stability.

        Args:
            params: Dict of active parameter name → tensor.
            grads:  Dict of pseudo-gradient name → tensor.
        """
        with torch.no_grad():
            for name, param in params.items():
                grad = grads[name]

                # Clip gradient for stability
                max_grad_norm = 10.0
                grad_norm = grad.norm()
                if grad_norm > max_grad_norm:
                    grad = grad * (max_grad_norm / grad_norm)

                # Momentum: accumulate moving average
                if self.momentum > 0:
                    if name not in self.momentum_buffer:
                        self.momentum_buffer[name] = torch.zeros_like(param)
                    buf = self.momentum_buffer[name]
                    buf.mul_(self.momentum).add_(grad, alpha=1.0 - self.momentum)
                    param.data.sub_(self.lr * buf)
                else:
                    param.data.sub_(self.lr * grad)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. With SPSA, uses only 2 forward passes regardless of model size.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.

        Returns:
            The loss value at the *start* of the step (before any update).

        Note:
            With SPSA enabled, this uses only 2 forward passes per step.
            Without SPSA, it uses 2 * num_active_parameters forward passes.
        """
        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        self.step_count += 1
        return float(loss_before)
