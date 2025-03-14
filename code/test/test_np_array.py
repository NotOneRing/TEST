import unittest
import numpy as np


class TestNumpyArray(unittest.TestCase):
    """Test cases for numpy array creation and properties."""

    def test_array_creation_and_shape(self):
        """Test creating a numpy array and verifying its shape."""
        # Define the maximum standard deviation
        std_max = 2
        
        # Create a numpy array with a single element (std_max squared)
        arr = np.array([std_max**2])
        
        # Verify the array contains the expected value
        self.assertEqual(arr[0], 4)
        
        # Verify the array has the expected shape (1,)
        self.assertEqual(arr.shape, (1,))
        
        # # Print statements for debugging (can be removed in production code)
        # print("arr = ", arr)
        # print("arr.shape = ", arr.shape)


if __name__ == '__main__':
    unittest.main()
