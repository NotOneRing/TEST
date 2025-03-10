import unittest
import tensorflow as tf
import numpy as np

class TestTensorflowOperations(unittest.TestCase):
    
    def test_reshape_linspace(self):
        """Test the reshape and linspace operations."""
        # Create the tensor as in the original file
        a = tf.reshape(tf.linspace(0, 20, 20), (2, 2, 5))
        
        # Check the shape
        self.assertEqual(a.shape, (2, 2, 5))
        
        # Check the values (approximately)
        expected_values = np.linspace(0, 20, 20).reshape(2, 2, 5)
        np.testing.assert_allclose(a.numpy(), expected_values, rtol=1e-5)
    
    def test_range(self):
        """Test the range operation."""
        # Create the tensor as in the original file
        b = tf.range(start=0, limit=5, delta=2)
        
        # Check the shape and values
        self.assertEqual(b.shape, (3,))
        np.testing.assert_array_equal(b.numpy(), np.array([0, 2, 4]))
    
    def test_gather(self):
        """Test the gather operation."""
        # Create the tensors as in the original file
        a = tf.reshape(tf.linspace(0, 20, 20), (2, 2, 5))
        b = tf.range(start=0, limit=5, delta=2)
        
        # Perform the gather operation
        c = tf.gather(a, b, axis=2)
        
        # Check the shape
        self.assertEqual(c.shape, (2, 2, 3))
        
        # Check the values
        # The gather operation should select elements at indices 0, 2, 4 along axis 2
        expected = tf.gather(a, [0, 2, 4], axis=2)
        np.testing.assert_allclose(c.numpy(), expected.numpy())
    
    def test_cuda_availability(self):
        """Test that CUDA information can be retrieved without errors."""
        # This test just ensures these calls don't raise exceptions
        try:
            is_cuda = tf.test.is_built_with_cuda()
            self.assertIsInstance(is_cuda, bool)
            
            gpu_devices = tf.config.list_physical_devices('GPU')
            self.assertIsInstance(gpu_devices, list)
            
            build_info = tf.sysconfig.get_build_info()
            self.assertIsInstance(build_info, dict)
            
            # Only test these if build_info contains these keys
            if 'cuda_version' in build_info:
                self.assertIsInstance(build_info['cuda_version'], str)
            if 'cudnn_version' in build_info:
                self.assertIsInstance(build_info['cudnn_version'], str)
                
        except Exception as e:
            self.fail(f"CUDA availability check raised exception: {e}")

if __name__ == '__main__':
    unittest.main()

















