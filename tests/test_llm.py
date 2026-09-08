import velocityai as vai
from velocityai.nn.transformer import LlamaModel
import numpy as np

def test_toy_llama():
    print("Testing VelocityAI Toy Llama Architecture...")
    
    # Toy Llama configs
    vocab_size = 128
    dim = 64
    num_heads = 4
    hidden_dim = 128
    num_layers = 2
    
    # Init model
    model = LlamaModel(vocab_size, dim, num_heads, hidden_dim, num_layers)
    
    # Dummy input: Batch=2, SequenceLength=10
    idx = vai.tensor(np.random.randint(0, vocab_size, (2, 10)).astype(np.int32))
    
    print("Running Forward Pass...")
    out = model(idx)
    
    assert out.shape == (2, 10, vocab_size), "Output shape is incorrect!"
    print(f"Forward Pass Success! Output Shape: {out.shape}")
    print("Transformer Block & LLM Architecture PASSED!")

if __name__ == "__main__":
    test_toy_llama()
