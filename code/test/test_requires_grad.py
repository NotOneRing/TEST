import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import numpy as np
from util.torch_to_tf import torch_tensor_requires_grad_

class TestRequiresGrad(unittest.TestCase):
    def setUp(self):
        # PyTorch network
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(2, 2)
                self.fc2 = nn.Linear(2, 1)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        # TensorFlow network
        class SimpleNet_tf(tf.keras.Model):
            def __init__(self):
                super(SimpleNet_tf, self).__init__()
                self.dense1 = tf.keras.layers.Dense(2)
                self.dense2 = tf.keras.layers.Dense(1)
            
            def call(self, inputs):
                x = self.dense1(inputs)
                return self.dense2(x)
        
        self.SimpleNet = SimpleNet
        self.SimpleNet_tf = SimpleNet_tf

    def test_pytorch_requires_grad(self):
        """Test PyTorch requires_grad functionality"""
        # create a PyTorch tensor
        tensor_pytorch = torch.tensor([1.0, 2.0], requires_grad=False)
        
        # use the requires_grad_() method to set gradient calculation
        tensor_pytorch.requires_grad_()  # set tensor_pytorch to have gradient calculation
        
        # check if the tensor needs to have gradient calculation
        self.assertTrue(tensor_pytorch.requires_grad)

    def test_pytorch_network_gradients(self):
        """Test PyTorch network gradients"""
        # create a PyTorch network and train
        net = self.SimpleNet()
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        
        input_tensor = torch.tensor([1.0, 2.0], requires_grad=False)
        self.assertEqual(input_tensor.shape, torch.Size([2]))
        
        output = net(input_tensor)
        
        # backward propagation
        output.backward()
        
        # check gradients
        self.assertIsNone(input_tensor.grad)  # Should be None since requires_grad=False
        
        # check that gradients of the networks' parameters exist
        for name, param in net.named_parameters():
            self.assertIsNotNone(param.grad)
        
        return net  # Return for comparison with TensorFlow

    def test_tensorflow_trainable(self):
        """Test TensorFlow trainable functionality"""
        # create a TensorFlow variable
        tensor_tf = tf.Variable([1.0, 2.0], trainable=False)
        
        # check if the tensor needs the gradient calculation
        self.assertFalse(tensor_tf.trainable)

    def test_pytorch_tensorflow_gradient_comparison(self):
        """Test comparison between PyTorch and TensorFlow gradients"""
        # Get PyTorch network with gradients
        net = self.test_pytorch_network_gradients()
        
        # create a TensorFlow network and train
        net_tf = self.SimpleNet_tf()
        
        # Initialize with random weights to ensure the model is built
        _ = net_tf(tf.constant(np.random.randn(1, 2).astype(np.float32)))
        
        # Copy weights from PyTorch to TensorFlow
        net_tf.dense1.trainable_weights[0].assign(net.fc1.weight.detach().numpy().T)  # kernel
        net_tf.dense1.trainable_weights[1].assign(net.fc1.bias.detach().numpy())     # bias
        
        net_tf.dense2.trainable_weights[0].assign(net.fc2.weight.detach().numpy().T)  # kernel
        net_tf.dense2.trainable_weights[1].assign(net.fc2.bias.detach().numpy())     # bias
        
        # Create input tensor for TensorFlow
        input_tensor_tf = tf.Variable([1.0, 2.0], trainable=True)
        input_tensor_tf = torch_tensor_requires_grad_(input_tensor_tf, requires_grad=False)
        
        self.assertEqual(input_tensor_tf.shape, tf.TensorShape([2]))
        
        # Compute gradients using GradientTape
        with tf.GradientTape(persistent=True) as tape:
            input_tensor_tf = tf.reshape(input_tensor_tf, [1, 2])
            output = net_tf(input_tensor_tf)
        
        # Get gradients for input and network parameters
        grads0 = tape.gradient(output, input_tensor_tf)
        grads1 = tape.gradient(output, net_tf.dense1.trainable_weights[0])  # fc1.weight
        grads2 = tape.gradient(output, net_tf.dense1.trainable_weights[1])  # fc1.bias
        grads3 = tape.gradient(output, net_tf.dense2.trainable_weights[0])  # fc2.weight
        grads4 = tape.gradient(output, net_tf.dense2.trainable_weights[1])  # fc2.bias
        
        # Get PyTorch gradients for comparison
        torch_params = dict(net.named_parameters())
        
        # Compare gradients
        # For input tensor, both should be None since requires_grad=False
        self.assertIsNone(grads0)
        
        # For network parameters
        tf_grads1 = grads1.numpy()
        torch_grads1 = torch_params['fc1.weight'].grad.detach().numpy()
        self.assertTrue(np.allclose(tf_grads1, torch_grads1.T))
        
        self.assertTrue(np.allclose(
            grads2.numpy(), 
            torch_params['fc1.bias'].grad.detach().numpy()
        ))
        
        self.assertTrue(np.allclose(
            grads3.numpy(), 
            torch_params['fc2.weight'].grad.detach().numpy().T
        ))
        
        self.assertTrue(np.allclose(
            grads4.numpy(), 
            torch_params['fc2.bias'].grad.detach().numpy()
        ))
        
        del tape

if __name__ == '__main__':
    unittest.main()
