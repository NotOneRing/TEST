import unittest
import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_tensor_detach


class PyTorchNet(torch.nn.Module):
    """A simple PyTorch neural network."""
    def __init__(self):
        super(PyTorchNet, self).__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TensorFlowNet(tf.keras.Model):
    """A simple TensorFlow neural network."""
    def __init__(self):
        super(TensorFlowNet, self).__init__()
        self.fc = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.fc(x)


class TestDetach(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Define a simple neural network in PyTorch
        self.pytorch_net = PyTorchNet()
        
        # Define a similar network in TensorFlow
        self.tensorflow_net = TensorFlowNet()
        
        # Build the TensorFlow model with input shape
        self.tensorflow_net.build((None, 4))
        
        # Initialize model
        _ = self.tensorflow_net(tf.constant(np.random.randn(1, 4).astype(np.float32)))
        
        # Copy weights from PyTorch to TensorFlow
        self.copy_weights_from_pytorch_to_tensorflow()
        
        # Generate input data
        self.input_data = np.random.rand(3, 4).astype(np.float32)

    def copy_weights_from_pytorch_to_tensorflow(self):
        """Copy weights from PyTorch model to TensorFlow model."""
        pytorch_weights = self.pytorch_net.state_dict()
        for i, weight in enumerate(self.tensorflow_net.trainable_variables):
            key = list(pytorch_weights.keys())[i]
            temp_weight = pytorch_weights[key].detach().numpy().T if 'weight' in key else pytorch_weights[key].detach().numpy()
            weight.assign(temp_weight)

    def test_model_params_equality(self):
        """Test if the parameters of both models are the same after copying."""
        self.assertTrue(self.compare_model_params(), "The model parameters should be the same")

    def test_detach_output(self):
        """Test if detach operation produces the same output in PyTorch and TensorFlow."""
        # Run multiple epochs
        num_epochs = 5
        for epoch in range(num_epochs):
            # PyTorch forward pass and detach
            pytorch_input = torch.tensor(self.input_data, requires_grad=True)
            pytorch_output = self.pytorch_net(pytorch_input)
            pytorch_detached_output = pytorch_output.detach().numpy()

            # TensorFlow forward pass and stop_gradient
            tensorflow_input = tf.convert_to_tensor(self.input_data)
            tensorflow_output = self.tensorflow_net(tensorflow_input)
            tensorflow_detached_output = torch_tensor_detach(tensorflow_output).numpy()

            # print("pytorch_detached_output = ", pytorch_detached_output)
            # print("tensorflow_detached_output = ", tensorflow_detached_output)

            # Check if the outputs are the same
            self.assertTrue(
                np.allclose(pytorch_detached_output, tensorflow_detached_output, atol=1e-3),
                f"Outputs differ in epoch {epoch + 1}"
            )

    def compare_model_params(self):
        """Compare the parameters of PyTorch and TensorFlow models."""
        pytorch_params = self.pytorch_net.state_dict()
        tensorflow_params = self.tensorflow_net.trainable_variables

        for i, tensorflow_param in enumerate(tensorflow_params):
            # Compare the value of each parameter
            key = list(pytorch_params.keys())[i]
            temp = pytorch_params[key]

            if 'weight' in key:
                temp = temp.detach().numpy().T
            else:
                temp = temp.detach().numpy()
            
            if not np.allclose(temp, tensorflow_param.numpy(), atol=1e-4):
                return False
        return True



if __name__ == '__main__':
    unittest.main()
