import tensorflow as tf
import torch
import numpy as np

from util.torch_to_tf import nn_Dropout


# torch.manual_seed(42)
# tf.random.set_seed(42)

# test code
if __name__ == "__main__":
    # initialize parameters
    input_tensor = np.random.rand(4, 4).astype(np.float32)
    dropout_prob = 0.5  # dropout rate

    # PyTorch part
    torch_dropout = torch.nn.Dropout(p=dropout_prob)
    torch_input = torch.tensor(input_tensor, requires_grad=True)
    torch_output = torch_dropout(torch_input)

    # TensorFlow part
    tf_dropout = nn_Dropout(p=dropout_prob)
    tf_input = tf.convert_to_tensor(input_tensor)
    tf_output = tf_dropout(tf_input, training=True)

    # compare results
    print("Input Tensor:")
    print(input_tensor)
    print("\nPyTorch Output:")
    print(torch_output.detach().numpy())
    print("\nTensorFlow Output:")
    print(tf_output.numpy())

    # check if it satisfies dropout's behavior: the ratio of zeros should be close to dropout_prob
    torch_zeros_ratio = np.mean(torch_output.detach().numpy() == 0)
    tf_zeros_ratio = np.mean(tf_output.numpy() == 0)
    print(f"\nPyTorch zeros ratio: {torch_zeros_ratio:.2f}")
    print(f"TensorFlow zeros ratio: {tf_zeros_ratio:.2f}")






















