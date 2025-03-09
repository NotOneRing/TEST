import pytest
import tensorflow as tf



# sampling.py FINISHED

# model/diffusion/sampling.py
import numpy as np
from model.diffusion.sampling import cosine_beta_schedule  

def test_cosine_beta_schedule():
    timesteps = 10
    s = 0.008
    dtype = tf.float32

    # call function
    betas = cosine_beta_schedule(timesteps, s, dtype)

    # check type of betas
    assert isinstance(betas, tf.Tensor), "Output is not a tf.Tensor"

    # chcek dtype of betas
    assert betas.dtype == dtype, f"Expected dtype {dtype}, but got {betas.dtype}"

    # check the length of betas
    assert betas.shape[0] == timesteps, f"Expected length {timesteps}, but got {betas.shape[0]}"

    # check beta's value within [0, 0.999]
    assert tf.reduce_all((betas >= 0) & (betas <= 0.999)), "Betas are out of expected range [0, 0.999]"

    # check betas and expected_betas_clipped are close
    alphas_cumprod = np.cos(((np.linspace(0, timesteps + 1, timesteps + 1) / (timesteps + 1)) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    expected_betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    expected_betas_clipped = np.clip(expected_betas, a_min=0, a_max=0.999)

    assert tf.reduce_all(tf.abs(betas - tf.convert_to_tensor(expected_betas_clipped, dtype=dtype)) < 1e-6), \
        "Betas do not match the expected values"



from model.diffusion.sampling import extract

def test_extract():
    # Create test inputs
    a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)  # Shape: (2, 3)
    t = tf.constant([0, 2], dtype=tf.int32)  # Shape: (2,)
    x_shape = (2, 1, 1)

    # Call the function
    output = extract(a, t, x_shape)

    # Expected output
    expected_output = tf.constant([[[1]], [[6]]], dtype=tf.float32)

    # Assertions
    assert output.shape == (2, 1, 1), "Output shape is incorrect."
    tf.debugging.assert_near(output, expected_output, message="Output values are incorrect.")
    print("Test passed!")




from model.diffusion.sampling import make_timesteps

def test_make_timesteps():
    batch_size = 4
    i = 10
    device = "/GPU:0"  # Ensure the test is running on GPU
    
    # Create a strategy to set the device
    with tf.device(device):
        t = make_timesteps(batch_size, i, device)
    
    assert isinstance(t, tf.Tensor), "Output is not a tf.Tensor"
    assert t.dtype == tf.int64, f"Expected dtype tf.int64, but got {t.dtype}"
    assert t.shape == (batch_size,), f"Expected shape {(batch_size,)}, but got {t.shape}"
    assert tf.reduce_all(tf.equal(t, i)), f"All elements in the tensor should be {i}, but got {t.numpy()}"
    
    # Modify the device check to compare only the device type (CPU or GPU)
    assert ('GPU' in t.device) == ('GPU' in device), f"Expected device {device}, but got {t.device}"



@pytest.mark.parametrize("batch_size, i, device", [
    (1, 0, "/CPU:0"),
    (5, 42, "/CPU:0"),
    (10, 7, "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0")
])
def test_make_timesteps_parametrized(batch_size, i, device):
    with tf.device(device):
        t = make_timesteps(batch_size, i, device)

    assert isinstance(t, tf.Tensor), "Output is not a tf.Tensor"
    assert t.dtype == tf.int64, f"Expected dtype tf.int64, but got {t.dtype}"
    assert t.shape == (batch_size,), f"Expected shape {(batch_size,)}, but got {t.shape}"
    assert tf.reduce_all(tf.equal(t, i)), f"All elements in the tensor should be {i}, but got {t.numpy()}"
    
    # Modify the device check to compare only the device type (CPU or GPU)
    assert ('GPU' in t.device) == ('GPU' in device), f"Expected device {device}, but got {t.device}"








