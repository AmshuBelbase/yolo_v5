import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Specify CUDA device
    device = torch.device('cuda')
    
    # Create a tensor on CUDA device
    cuda_tensor = torch.tensor([1, 2, 3], device=device)
    
    # Perform operations on CUDA tensor directly
    result = cuda_tensor * 2
    
    # Print result directly (no need to move to CPU)
    print(result)
else:
    print('CUDA (GPU) is not available. Please check your system configuration.')
