import numpy as np
import torch
import torch_neuronx
from torch import nn
import neuronx.tracer as tracer

class TrainiumMatMul(nn.Module):
    """
    Optimized matrix multiplication for AWS Trainium architecture.
    Implements tiled matrix multiplication optimized for NeuronCore clusters
    with specific attention to Trainium's tensor compute units (TCUs).
    """
    def __init__(self, transpose_b=False, dtype=torch.float32):
        super().__init__()
        self.transpose_b = transpose_b
        self.dtype = dtype
        
        # Trainium-specific constants
        self.NEURON_CORE_COUNT = 2  # Per NeuronCore cluster
        self.TCU_TILE_SIZE = 32     # Trainium's TCU operates on 32x32 tiles
        self.BF16_SUPPORTED = True  # Trainium supports BF16 natively
        
    def forward(self, matrix_a, matrix_b):
        """
        Forward pass implementing matrix multiplication.
        
        Args:
            matrix_a: First input matrix of shape (M, K)
            matrix_b: Second input matrix of shape (K, N) or (N, K) if transpose_b
            
        Returns:
            Result matrix of shape (M, N)
        """
        # Convert inputs to appropriate dtype
        if self.dtype == torch.bfloat16 and self.BF16_SUPPORTED:
            matrix_a = matrix_a.to(torch.bfloat16)
            matrix_b = matrix_b.to(torch.bfloat16)
        
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
        
        # Create padded matrices
        a_padded = self._pad_matrix(matrix_a, (M_padded, K_padded))
        b_padded = self._pad_matrix(matrix_b, (K_padded, N_padded))
        
        # Perform tiled matrix multiplication
        result = self._tiled_matmul(a_padded, b_padded)
        
        # Remove padding from result
        return result[:M, :N]
    
    def _pad_dimension(self, dim):
        """Pad dimension to TCU tile boundary"""
        return ((dim + self.TCU_TILE_SIZE - 1) // self.TCU_TILE_SIZE) * self.TCU_TILE_SIZE
    
    def _pad_matrix(self, matrix, padded_shape):
        """Pad matrix with zeros to match TCU tile size"""
        result = torch.zeros(padded_shape, dtype=matrix.dtype, device=matrix.device)
        result[:matrix.shape[0], :matrix.shape[1]] = matrix
        return result
    
    def _tiled_matmul(self, a_padded, b_padded):
        """
        Performs tiled matrix multiplication optimized for Trainium's TCUs.
        Uses implicit GEMM decomposition for better hardware utilization.
        """
        M, K = a_padded.shape
        K, N = b_padded.shape
        
        # Initialize result matrix
        result = torch.zeros((M, N), dtype=a_padded.dtype, device=a_padded.device)
        
        # Perform tiled multiplication
        for i in range(0, M, self.TCU_TILE_SIZE):
            for j in range(0, N, self.TCU_TILE_SIZE):
                for k in range(0, K, self.TCU_TILE_SIZE):
                    # Extract tiles
                    a_tile = a_padded[i:i+self.TCU_TILE_SIZE, k:k+self.TCU_TILE_SIZE]
                    b_tile = b_padded[k:k+self.TCU_TILE_SIZE, j:j+self.TCU_TILE_SIZE]
                    
                    # Perform tile multiplication using TCU
                    tile_result = torch.matmul(a_tile, b_tile)
                    
                    # Accumulate result
                    result[i:i+self.TCU_TILE_SIZE, j:j+self.TCU_TILE_SIZE] += tile_result
        
        return result

def compile_for_trainium(model, example_inputs):
    """
    Compiles the model for Trainium using torch-neuronx.
    
    Args:
        model: TrainiumMatMul model instance
        example_inputs: Tuple of example input tensors
        
    Returns:
        Compiled model
    """
    # Set up Trainium-specific options
    compile_options = {
        "tensor_parallel_degree": 2,  # Use 2 NeuronCores per operation
        "enable_fast_loading": True,
        "enable_broadcasting": True,
        "enable_experimental_optimizations": True
    }
    
    # Compile model for Trainium
    return torch_neuronx.trace(
        model,
        example_inputs,
        compiler_options=compile_options
    )

# Example usage
if __name__ == "__main__":
    # Create sample matrices
    M, K, N = 128, 64, 32
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    
    # Create and compile model
    model = TrainiumMatMul(dtype=torch.bfloat16)
    example_inputs = (A, B)
    
    # Compile for Trainium
    compiled_model = compile_for_trainium(model, example_inputs)
    
    # Run inference
    with torch.no_grad():
        result = compiled_model(A, B)
    
    # Verify result
    expected = torch.matmul(A, B)
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
    print("Matrix multiplication successful!")
