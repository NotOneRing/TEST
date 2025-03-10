
import unittest
import torch
import numpy as np

class TestPyTorchOperations(unittest.TestCase):
    
    def setUp(self):
        """Set up common test variables."""
        self.a = torch.tensor([0], dtype=torch.int64)
        self.b = torch.linspace(0, 20, 20).reshape(2, 2, 5)
        self.c = torch.linspace(0, 20, 20).reshape(2, 2, 5)
    
    def test_cuda_availability(self):
        """Test CUDA availability information."""
        # Just verify these calls don't raise exceptions
        version = torch.version.__version__
        self.assertIsInstance(version, str)
        
        is_available = torch.cuda.is_available()
        self.assertIsInstance(is_available, bool)
        
        device_count = torch.cuda.device_count()
        self.assertIsInstance(device_count, int)
        
        # Only test if CUDA is available
        if is_available and device_count > 0:
            current_device = torch.cuda.current_device()
            self.assertIsInstance(current_device, int)
    
    def test_gpu_tensor_creation(self):
        """Test creating tensors on GPU and moving to CPU."""
        # Skip test if CUDA is not available
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            self.skipTest("CUDA not available, skipping GPU tensor test")
        
        # Create tensor on GPU
        tensor_gpu = torch.rand((3, 3), device='cuda:0')
        self.assertEqual(tensor_gpu.device.type, 'cuda')
        self.assertEqual(tensor_gpu.shape, (3, 3))
        
        # Move tensor to CPU
        tensor_cpu = tensor_gpu.cpu()
        self.assertEqual(tensor_cpu.device.type, 'cpu')
        self.assertEqual(tensor_cpu.shape, (3, 3))
    
    def test_tensor_creation(self):
        """Test tensor creation operations."""
        # Test tensor creation
        self.assertEqual(self.a.shape, (1,))
        self.assertEqual(self.a.dtype, torch.int64)
        self.assertEqual(self.a.item(), 0)
        
        # Test linspace and reshape
        self.assertEqual(self.b.shape, (2, 2, 5))
        # Check first and last values
        self.assertAlmostEqual(self.b[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(self.b[1, 1, 4].item(), 20.0)
    
    def test_tensor_indexing(self):
        """Test tensor indexing operations."""
        # Test indexing with tensor
        b_indexed = self.b[self.a]
        self.assertEqual(b_indexed.shape, (1, 2, 5))
        # The first element should be b[0]
        self.assertTrue(torch.allclose(b_indexed[0], self.b[0]))
        
        # Test indexing with ellipsis
        b_indexed_ellipsis = self.b[self.a, ...]
        self.assertEqual(b_indexed_ellipsis.shape, (1, 2, 5))
        self.assertTrue(torch.allclose(b_indexed_ellipsis[0], self.b[0]))
        
        # Test simple indexing
        c_indexed = self.c[0]
        self.assertEqual(c_indexed.shape, (2, 5))
        self.assertTrue(torch.allclose(c_indexed, self.c[0]))
        
        # Test tensor indexing
        c_indexed_tensor = self.c[self.a]
        self.assertEqual(c_indexed_tensor.shape, (1, 2, 5))
        self.assertTrue(torch.allclose(c_indexed_tensor[0], self.c[0]))

if __name__ == '__main__':
    unittest.main()
