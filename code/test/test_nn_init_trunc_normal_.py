import unittest
import tensorflow as tf
from util.torch_to_tf import torch_nn_init_trunc_normal_


class TestTruncNormalInitialization(unittest.TestCase):
    """
    Test case for truncated normal initialization function.
    This tests the torch_nn_init_trunc_normal_ function which initializes
    tensor values using a truncated normal distribution.
    """

    def test_trunc_normal_initialization(self):
        """
        Test that torch_nn_init_trunc_normal_ properly initializes a tensor
        with values from a truncated normal distribution with specified parameters.
        """
        # Create a test tensor with uniform values between 10 and 20
        tensor = tf.Variable(tf.random.uniform(shape=(30, 30), minval=10, maxval=20, dtype=tf.float32))
        
        # Store original values for comparison
        original_tensor = tensor.numpy().copy()
        
        # Apply truncated normal initialization
        torch_nn_init_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
        
        # Get the initialized tensor values
        initialized_tensor = tensor.numpy()
        
        # Verify the tensor was modified
        self.assertFalse(
            tf.reduce_all(tf.equal(tensor, original_tensor)),
            "Tensor should be modified after initialization"
        )
        
        # Verify values are within the truncated range [-2.0, 2.0]
        self.assertTrue(
            tf.reduce_all(tf.greater_equal(tensor, -2.0)),
            "All values should be greater than or equal to -2.0"
        )
        self.assertTrue(
            tf.reduce_all(tf.less_equal(tensor, 2.0)),
            "All values should be less than or equal to 2.0"
        )
        
        # Verify the mean is approximately 0.0 (with some tolerance)
        mean = tf.reduce_mean(tensor)
        self.assertAlmostEqual(
            mean.numpy(), 0.0, delta=0.1,
            msg="Mean should be approximately 0.0"
        )
        
        # Verify the standard deviation is approximately 1.0 (with some tolerance)
        # Note: The actual std will be slightly less than 1.0 due to truncation
        std = tf.math.reduce_std(tensor)
        self.assertLess(
            abs(std.numpy() - 1.0), 0.3,
            msg="Standard deviation should be approximately 1.0"
        )


if __name__ == "__main__":
    unittest.main()
