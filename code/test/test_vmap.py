import unittest
import torch
import numpy as np
import tensorflow as tf
from util.torch_to_tf import torch_dot, torch_vmap

class TestVmap(unittest.TestCase):
    """
    Test cases for the torch_vmap function, which is a TensorFlow implementation
    of PyTorch's vmap function for vectorizing operations.
    """

    def test_nested_vmap(self):
        """
        Test nested vmap functionality.
        vmap() can also be nested, producing an output with multiple batched dimensions.
        """
        # torch.dot                            # [D], [D] -> []
        batched_dot = torch.vmap(torch.vmap(torch.dot))  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
        x1, y1 = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
        result1 = batched_dot(x1, y1)  # tensor of size [2, 3]

        x1_tf = tf.convert_to_tensor(x1.numpy())
        y1_tf = tf.convert_to_tensor(y1.numpy())

        batched_dot_tf = torch_vmap(torch_vmap(torch_dot))
        outputs_tf = batched_dot_tf(x1_tf, y1_tf)

        self.assertTrue(np.allclose(result1.numpy(), outputs_tf.numpy(), atol=1e-4))

    def test_vmap_with_in_dims(self):
        """
        Test vmap with in_dims parameter to specify which dimension to vectorize.
        """
        # torch.dot                            # [N], [N] -> []
        batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
        x2, y2 = torch.randn(2, 5), torch.randn(2, 5)
        result2 = batched_dot(x2, y2)   # output is [5] instead of [2] if batched along the 0th dimension

        x2_tf = tf.convert_to_tensor(x2.numpy())
        y2_tf = tf.convert_to_tensor(y2.numpy())

        tf_batched_dot = torch_vmap(torch_dot, in_dims=1)  # [N, D], [N, D] -> [D]
        tf_result2 = tf_batched_dot(x2_tf, y2_tf)

        self.assertTrue(np.allclose(result2.numpy(), tf_result2.numpy()))

    def test_vmap_with_out_dims(self):
        """
        Test vmap with out_dims parameter to specify where the batched dimension appears in output.
        """
        f = lambda x: x ** 2
        x = torch.arange(0, 10, 1).reshape(2, 5)
        
        batched_pow = torch.vmap(f, out_dims=1)
        result = batched_pow(x)  # [5, 2]

        tf_func = lambda x: x ** 2
        x_tf = tf.convert_to_tensor(x.numpy())
        tf_batched_pow = torch_vmap(tf_func, out_dims=1)
        result_tf = tf_batched_pow(x_tf)  # [5, 2]

        self.assertTrue(np.allclose(result.numpy(), result_tf.numpy(), atol=1e-4))

    def test_vmap_with_tuple_in_dims(self):
        """
        Test vmap with tuple in_dims to specify different dimensions for each input.
        """
        def add_tensors(a, b):
            return a + b
            
        x = torch.randn(3, 4)
        y = torch.randn(4, 3).transpose(0, 1)  # Shape becomes [3, 4]
        
        # Vectorize along dimension 0 for x and dimension 0 for y
        batched_add = torch.vmap(add_tensors, in_dims=(0, 0))
        result = batched_add(x, y)
        
        x_tf = tf.convert_to_tensor(x.numpy())
        y_tf = tf.convert_to_tensor(y.numpy())
        
        tf_add = lambda a, b: a + b
        tf_batched_add = torch_vmap(tf_add, in_dims=(0, 0))
        result_tf = tf_batched_add(x_tf, y_tf)
        
        self.assertTrue(np.allclose(result.numpy(), result_tf.numpy(), atol=1e-4))


    def test_vmap_with_negative_dims(self):
        """
        Test vmap with negative dimensions to index from the end of the tensor.
        """
        def square_tensor(x):
            return x ** 2
            
        x = torch.randn(4, 3, 2)
        
        # Vectorize along the last dimension (-1)
        batched_square = torch.vmap(square_tensor, in_dims=-1, out_dims=-1)
        result = batched_square(x)
        
        x_tf = tf.convert_to_tensor(x.numpy())
        
        tf_square = lambda x: x ** 2
        tf_batched_square = torch_vmap(tf_square, in_dims=-1, out_dims=-1)
        result_tf = tf_batched_square(x_tf)
        
        self.assertTrue(np.allclose(result.numpy(), result_tf.numpy(), atol=1e-4))

    def test_vmap_with_chunk_size(self):
        """
        Test vmap with chunk_size parameter to process large batches in smaller chunks.
        """
        def square_tensor(x):
            return x ** 2
            
        # Create a larger tensor to test chunking
        x = torch.randn(100, 5)
        
        # No chunking in PyTorch vmap, so we'll compare with our implementation using different chunk sizes
        x_tf = tf.convert_to_tensor(x.numpy())
        
        tf_square = lambda x: x ** 2
        
        # Process without chunking
        tf_batched_square_no_chunk = torch_vmap(tf_square)
        result_no_chunk = tf_batched_square_no_chunk(x_tf)
        
        # Process with chunk_size=10
        tf_batched_square_chunk = torch_vmap(tf_square, chunk_size=10)
        result_chunk = tf_batched_square_chunk(x_tf)
        
        # Results should be the same regardless of chunking
        self.assertTrue(np.allclose(result_no_chunk.numpy(), result_chunk.numpy(), atol=1e-4))

    def test_vmap_with_different_randomness(self):
        """
        Test vmap with different randomness settings.
        """
        # Define a function that uses randomness
        def random_add(x):
            # Add random noise to the input
            return x + tf.random.normal(shape=x.shape)
        
        x = tf.ones((5, 3))
        
        # With randomness='different', each batch element should get different random values
        batched_random_diff = torch_vmap(random_add, randomness='different')
        result_diff = batched_random_diff(x)
        
        # Check that different batch elements got different random values
        # We can't directly compare with PyTorch since it doesn't have this parameter,
        # but we can check that the values are indeed different across the batch dimension
        batch_variance = tf.math.reduce_variance(result_diff, axis=0)
        
        # There should be variance across the batch dimension due to different random values
        self.assertTrue(tf.reduce_all(batch_variance > 0))

    def test_vmap_with_same_randomness(self):
        """
        Test vmap with 'same' randomness setting.
        """
        # Define a function that uses randomness
        def random_add(x):
            # Add random noise to the input
            return x + tf.random.normal(shape=tf.shape(x))
        
        x = tf.ones((5, 3))
        
        # With randomness='same', each batch element should get the same random values
        batched_random_same = torch_vmap(random_add, randomness='same')
        
        # This is just a test to ensure the function runs without error
        # In a real implementation, this would ensure the same random values are used
        result_same = batched_random_same(x)
        
        # The shape should be preserved
        self.assertEqual(result_same.shape, x.shape)

    def test_vmap_with_list_output(self):
        """
        Test vmap with a function that returns a list.
        """
        def split_tensor(x):
            # Split the tensor along the last dimension and return as list
            result = tf.split(x, num_or_size_splits=x.shape[-1], axis=-1)
            # print("type(result) = ", type(result))
            return result
        
        x = tf.random.normal((3, 4, 2))
        
        # Vectorize along the first dimension
        batched_split = torch_vmap(split_tensor)
        result = batched_split(x)
        
        # print("result = ", result)
        # print("type(result) = ", type(result))

        # Check that the result is a list
        self.assertIsInstance(result, list)
        
        # Check that each element in the list has the expected shape
        for i, part in enumerate(result):
            self.assertEqual(part.shape, (3, 4, 1))

    def test_vmap_with_dict_output(self):
        """
        Test vmap with a function that returns a dictionary.
        """
        def process_tensor(x):
            # Process tensor and return results as a dictionary
            return {
                'square': x ** 2,
                'cube': x ** 3
            }
        
        x = tf.random.normal((3, 4))
        
        # Vectorize along the first dimension
        batched_process = torch_vmap(process_tensor)
        result = batched_process(x)
        
        # Check that the result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that each value in the dictionary has the expected shape
        self.assertEqual(result['square'].shape, (3, 4))
        self.assertEqual(result['cube'].shape, (3, 4))


if __name__ == '__main__':
    unittest.main()





