#!/usr/bin/env python3
"""
Quick test to verify SPSA optimizer works correctly.
"""

import torch
import torch.nn as nn
from zo_optimizer import ZeroOrderOptimizer

def test_spsa_optimizer():
    """Test that SPSA optimizer can be instantiated and step through."""
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    
    # Initialize optimizer with SPSA enabled
    optimizer = ZeroOrderOptimizer(
        model,
        lr=0.01,
        eps=1e-3,
        use_spsa=True,
        momentum=0.9,
    )
    
    # Override layer_names to tune a simple layer
    optimizer.layer_names = ["2.weight", "2.bias"]
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    
    # Define loss function
    loss_fn_module = nn.CrossEntropyLoss()
    
    def loss_fn():
        logits = model(x)
        loss = loss_fn_module(logits, y)
        return float(loss)
    
    # Test that step works
    print("Testing SPSA optimizer...")
    for i in range(5):
        loss = optimizer.step(loss_fn)
        print(f"Step {i+1}, Loss: {loss:.4f}")
    
    print("\nSPSA optimizer test passed!")
    print(f"Step count: {optimizer.step_count}")
    print(f"Momentum buffer keys: {list(optimizer.momentum_buffer.keys())}")
    

if __name__ == "__main__":
    test_spsa_optimizer()
