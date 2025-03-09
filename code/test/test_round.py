import torch
import tensorflow as tf

from util.torch_to_tf import torch_round

import numpy as np

def test_round():
    # Create a tensor of float type
    tensor = torch.tensor([1.1, 2.5, 3.7, -1.4])

    # round off
    rounded_tensor = torch.round(tensor)

    print(rounded_tensor)





    # Create a tensor of float type
    tensor = tf.constant([1.1, 2.5, 3.7, -1.4])

    # round off
    tf_rounded_tensor = torch_round(tensor)

    print(tf_rounded_tensor)


    assert np.allclose(rounded_tensor, tf_rounded_tensor)




test_round()

