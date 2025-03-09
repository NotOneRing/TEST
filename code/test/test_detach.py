import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_tensor_detach

# Define a simple neural network in PyTorch
class PyTorchNet(torch.nn.Module):
    def __init__(self):
        super(PyTorchNet, self).__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

# Define a similar network in TensorFlow
class TensorFlowNet(tf.keras.Model):
    def __init__(self):
        super(TensorFlowNet, self).__init__()
        self.fc = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.fc(x)

# Initialize the PyTorch model and TensorFlow model with the same weights
pytorch_net = PyTorchNet()
tensorflow_net = TensorFlowNet()

tensorflow_net.build((None, 4))  # input shape (batch_size, 4)
# print("tensorflow_net.trainable_variables = ", tensorflow_net.trainable_variables)

#initialize model
_ = tensorflow_net(tf.constant(np.random.randn(1, 4).astype(np.float32)))

print("tensorflow_net.trainable_variables = ", tensorflow_net.trainable_variables)


# Copy weights from PyTorch to TensorFlow
pytorch_weights = pytorch_net.state_dict()
for i, weight in enumerate(tensorflow_net.trainable_variables):
    key = list(pytorch_weights.keys())[i]
    print("key = ", key)
    temp_weight = pytorch_weights[ key ].detach().numpy().T if 'weight' in key else pytorch_weights[ key ].detach().numpy()

    print("i = ", i)
    print("weight = ", weight)
    print('temp_weight = ', temp_weight)

    print("weight.shape = ", weight.shape)
    print('temp_weight.shape = ', temp_weight.shape)

    weight.assign(temp_weight)






# compare network parameters
def compare_model_params(pytorch_model, tensorflow_model):
    # pytorch_params = list(pytorch_model.parameters())
    pytorch_params = pytorch_model.state_dict()
    
    tensorflow_params = tensorflow_model.trainable_variables

    for i, tensorflow_param in enumerate(tensorflow_params):
        # compare the value of each parameter
        key = list( pytorch_params.keys() )[i]
        temp = pytorch_weights[key]

        if 'weight' in key:
            temp = temp.detach().numpy().T
        else:
            temp = temp.detach().numpy()
        
        if not np.allclose(temp, tensorflow_param.numpy(), atol=1e-6):
            print(f"Parameters mismatch:\nPyTorch param: {temp}\nTensorFlow param: {tensorflow_param.numpy()}")
            return False
    return True





# call functino to check the parameters are equivalent
if compare_model_params(pytorch_net, tensorflow_net):
    print("The parameters are the same!")
else:
    print("The parameters are different!")



# Generate the same input data
input_data = np.random.rand(3, 4).astype(np.float32)




def test_detach():

    # Run multiple epochs
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # PyTorch forward pass and detach
        pytorch_input = torch.tensor(input_data, requires_grad=True)
        pytorch_output = pytorch_net(pytorch_input)
        pytorch_detached_output = pytorch_output.detach().numpy()

        # TensorFlow forward pass and stop_gradient
        tensorflow_input = tf.convert_to_tensor(input_data)
        tensorflow_output = tensorflow_net(tensorflow_input)
        tensorflow_detached_output = torch_tensor_detach(tensorflow_output).numpy()

        # Compare the outputs
        print("PyTorch detached output:\n", pytorch_detached_output)
        print("TensorFlow detached output:\n", tensorflow_detached_output)

        # Check if the outputs are the same
        if np.allclose(pytorch_detached_output, tensorflow_detached_output, atol=1e-6):
            print("The outputs are the same!")
        else:
            print("The outputs are different!")

        assert np.allclose(pytorch_detached_output, tensorflow_detached_output, atol=1e-6)

        print("-" * 50)





test_detach()

















 