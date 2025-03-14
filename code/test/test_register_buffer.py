import torch
import tensorflow as tf
import numpy as np
import unittest


# TensorFlow wrapper for registering buffers
from util.torch_to_tf import torch_register_buffer


# PyTorch class for testing torch.register_buffer
class TorchModel(torch.nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()

    def register_my_buffer(self, name, tensor):
        self.register_buffer(name, tensor)


# TensorFlow class for testing torch_register_buffer
class TFModel(tf.keras.layers.Layer):
    def __init__(self):
        super(TFModel, self).__init__()

    def register_my_buffer(self, name, tensor):
        torch_register_buffer(self, tensor, name)


class TestRegisterBuffer(unittest.TestCase):
    def test_register_buffer(self):
        """Test if torch_register_buffer works the same as torch.register_buffer"""
        # Input data
        buffer_name = "memory_mask"
        input_array = np.random.randn(5, 5).astype(np.float32)

        # PyTorch
        torch_model = TorchModel()
        torch_tensor = torch.tensor(input_array)
        torch_model.register_my_buffer(buffer_name, torch_tensor)
        torch_buffer = getattr(torch_model, buffer_name)

        # TensorFlow
        tf_model = TFModel()
        tf_model.register_my_buffer(buffer_name, input_array)
        tf_buffer = getattr(tf_model, buffer_name)

        # # Print outputs for debugging
        # print("PyTorch buffer:")
        # print(torch_buffer)

        # print("TensorFlow buffer:")
        # print(tf_buffer.numpy())

        # Check if values are close
        is_close = np.allclose(torch_buffer.numpy(), tf_buffer.numpy(), atol=1e-5)
        # print(f"Buffers match: {is_close}")

        # Use unittest assertion
        self.assertTrue(np.allclose(torch_buffer.numpy(), tf_buffer.numpy(), atol=1e-5),
                        "PyTorch and TensorFlow buffers should have the same values")


if __name__ == "__main__":
    unittest.main()
