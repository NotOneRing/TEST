import torch
import tensorflow as tf
import numpy as np

from util.torch_to_tf import torch_randint

# Set seeds for reproducibility
torch.manual_seed(42)
tf.random.set_seed(42)

def test_randint():
    # PyTorch randint
    low, high = 0, 100
    size = (2, 3)
    torch_tensor = torch.randint(low, high, size)

    # TensorFlow randint
    tf_tensor = torch_randint(low=low, high=high, size=size)

    # Compare outputs
    print("Torch tensor:\n", torch_tensor.numpy())
    print("TensorFlow tensor:\n", tf_tensor.numpy())
    print("Tensors match:", np.all(torch_tensor.numpy() == tf_tensor.numpy()))

# Run the test
if __name__ == "__main__":
    test_randint()
