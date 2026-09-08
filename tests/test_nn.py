import pytest
import numpy as np
import velocityai as vai

def test_nn_training_loop():
    # Model
    model = vai.nn.Sequential(
        vai.nn.Linear(10, 5),
        vai.nn.ReLU(),
        vai.nn.Linear(5, 2)
    )
    
    # Loss and optimizer
    loss_fn = vai.nn.CrossEntropyLoss()
    optimizer = vai.optim.Adam(model.parameters(), lr=0.1)
    
    # Data
    x = vai.randn(4, 10)
    y = vai.zeros((4,))  # Class 0
    
    initial_loss = None
    final_loss = None
    
    # Train for 5 epochs
    for epoch in range(5):
        optimizer.zero_grad()
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        if epoch == 0:
            initial_loss = loss.numpy()[0]
            
        loss.backward()
        optimizer.step()
        
        if epoch == 4:
            final_loss = loss.numpy()[0]
            
    assert final_loss < initial_loss
    
if __name__ == "__main__":
    pytest.main([__file__])
