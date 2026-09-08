import velocityai as vai
from velocityai.data import TensorDataset, DataLoader, transforms
from velocityai.distributed import DataParallel, CommGroup

def test_data_pipeline():
    print("Testing Data Pipeline...")
    # Create random data
    x = vai.randn(100, 10)
    y = vai.randn(100, 2)
    
    dataset = TensorDataset(x, y)
    assert len(dataset) == 100, "Dataset length incorrect"
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batches = list(loader)
    
    # 100 / 16 = 6.25 -> 7 batches
    assert len(batches) == 7, "DataLoader batches incorrect"
    
    # Check first batch shape
    bx, by = batches[0]
    assert bx.shape == (16, 10), "Batch x shape incorrect"
    assert by.shape == (16, 2), "Batch y shape incorrect"
    
    # Check last batch shape
    bx_last, by_last = batches[-1]
    assert bx_last.shape == (4, 10), "Last batch x shape incorrect"
    assert by_last.shape == (4, 2), "Last batch y shape incorrect"
    
    print("Data Pipeline PASSED!")
    
def test_distributed_stubs():
    print("Testing Distributed Stubs...")
    model = vai.nn.Linear(10, 5)
    
    # Wrap in DataParallel
    dp_model = DataParallel(model, device_ids=["cpu"])
    
    x = vai.randn(4, 10)
    out = dp_model(x)
    
    assert out.shape == (4, 5), "DataParallel forward pass incorrect shape"
    
    # Test CommGroup
    comm = CommGroup()
    comm.all_reduce(x) # Should just pass through safely
    
    print("Distributed Stubs PASSED!")

if __name__ == "__main__":
    test_data_pipeline()
    test_distributed_stubs()
