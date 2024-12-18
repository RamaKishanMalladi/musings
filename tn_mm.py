import numpy as np
import torch
import torch_neuronx
from torch import nn
import neuronx.tracer as tracer

class TrainiumMatMulFP32(nn.Module):
    """
    FP32-optimized matrix multiplication for AWS Trainium architecture.
    Implements tiled matrix multiplication optimized for NeuronCore clusters
    with specific attention to FP32 compute efficiency.
    """
    def __init__(self, transpose_b=False):
        super().__init__()
        self.transpose_b = transpose_b
        
        # Trainium-specific constants for FP32
        self.NEURON_CORE_COUNT = 2    # Per NeuronCore cluster
        self.TCU_TILE_SIZE = 16       # Smaller tiles for FP32 to maintain efficiency
        self.CACHE_LINE_SIZE = 64     # Trainium cache line size in bytes
        self.FP32_SIZE = 4            # Size of FP32 in bytes
        
    def forward(self, matrix_a, matrix_b):
        """
        Forward pass implementing FP32 matrix multiplication.
        
        Args:
            matrix_a: First input matrix of shape (M, K) in FP32
            matrix_b: Second input matrix of shape (K, N) or (N, K) if transpose_b
            
        Returns:
            Result matrix of shape (M, N) in FP32
        """
        # Ensure FP32 dtype
        matrix_a = matrix_a.to(torch.float32)
        matrix_b = matrix_b.to(torch.float32)
        
        # Handle matrix transposition if needed
        if self.transpose_b:
            matrix_b = matrix_b.transpose(-2, -1)
            
        # Pad matrices to tile boundaries for TCU efficiency
        M, K = matrix_a.shape
        K2, N = matrix_b.shape
        
        if K != K2:
            raise ValueError(f"Incompatible matrix dimensions: {matrix_a.shape} and {matrix_b.shape}")
            
        M_padded = self._pad_dimension(M)
        N_padded = self._pad_dimension(N)
        K_padded = self._pad_dimension(K)
        
        # Create padded matrices with cache line alignment
        a_padded = self._pad_and_align_matrix(matrix_a, (M_padded, K_padded))
        b_padded = self._pad_and_align_matrix(matrix_b, (K_padded, N_padded))
        
        # Perform tiled matrix multiplication
        result = self._tiled_matmul_fp32(a_padded, b_padded)
        
        # Remove padding from result
        return result[:M, :N]
    
    def _pad_dimension(self, dim):
        """
        Pad dimension to TCU tile boundary with consideration for FP32 alignment
        """
        # Ensure alignment with both TCU tile size and cache lines
        cache_elements = self.CACHE_LINE_SIZE // self.FP32_SIZE
        lcm = np.lcm(self.TCU_TILE_SIZE, cache_elements)
        return ((dim + lcm - 1) // lcm) * lcm
    
    def _pad_and_align_matrix(self, matrix, padded_shape):
        """
        Pad matrix with zeros and ensure proper memory alignment for FP32
        """
        # Create aligned storage
        result = torch.zeros(padded_shape, 
                           dtype=torch.float32,
                           device=matrix.device,
                           memory_format=torch.contiguous_format)
        
        # Copy data with proper alignment
        result[:matrix.shape[0], :matrix.shape[1]] = matrix
        return result
    
    def _tiled_matmul_fp32(self, a_padded, b_padded):
        """
        Performs tiled matrix multiplication optimized for Trainium's FP32 compute.
        Includes specific optimizations for FP32 precision.
        """
        M, K = a_padded.shape
        K, N = b_padded.shape
        
        # Initialize result matrix with proper alignment
        result = torch.zeros((M, N), 
                           dtype=torch.float32,
                           device=a_padded.device,
                           memory_format=torch.contiguous_format)
        
        # Perform tiled multiplication with FP32 optimizations
        for i in range(0, M, self.TCU_TILE_SIZE):
            for j in range(0, N, self.TCU_TILE_SIZE):
                accum = torch.zeros((self.TCU_TILE_SIZE, self.TCU_TILE_SIZE),
                                  dtype=torch.float32,
                                  device=result.device)
                
                for k in range(0, K, self.TCU_TILE_SIZE):
                    # Extract tiles
                    a_tile = a_padded[i:i+self.TCU_TILE_SIZE, k:k+self.TCU_TILE_SIZE]
                    b_tile = b_padded[k:k+self.TCU_TILE_SIZE, j:j+self.TCU_TILE_SIZE]
                    
                    # Perform FP32 tile multiplication
                    tile_result = torch.matmul(a_tile, b_tile)
                    accum += tile_result
                
                # Accumulate result with reduced numerical error
                result[i:i+self.TCU_TILE_SIZE, j:j+self.TCU_TILE_SIZE] = accum
        
        return result

def compile_for_trainium_fp32(model, example_inputs):
    """
    Compiles the model for Trainium with FP32-specific optimizations.
    
    Args:
        model: TrainiumMatMulFP32 model instance
        example_inputs: Tuple of example input tensors
        
    Returns:
        Compiled model
    """
    # Set up Trainium-specific options for FP32
    compile_options = {
        "tensor_parallel_degree": 2,
        "enable_fast_loading": True,
        "enable_broadcasting": True,
        "enable_experimental_optimizations": True,
        "auto_cast_type": "fp32",     # Force FP32 computation
        "preserve_precision": True,    # Maintain FP32 precision
        "optimization_level": 2        # Balance between speed and precision
    }
    
    # Compile model for Trainium
    return torch_neuronx.trace(
        model,
        example_inputs,
        compiler_options=compile_options
    )

# Example usage
if __name__ == "__main__":
    # Create sample matrices in FP32
    M, K, N = 128, 64, 32
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    
    # Create and compile model
    model = TrainiumMatMulFP32()
    example_inputs = (A, B)
    
    # Compile for Trainium
    compiled_model = compile_for_trainium_fp32(model, example_inputs)
    
    # Run inference
    with torch.no_grad():
        result = compiled_model(A, B)
    
    # Verify result with appropriate tolerance for FP32
    expected = torch.matmul(A, B)
    assert torch.allclose(result, expected, rtol=1e-6, atol=1e-6)
    print("FP32 matrix multiplication successful!")
