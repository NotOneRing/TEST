import unittest
import tensorflow as tf
import numpy as np
from model.diffusion.sampling import cosine_beta_schedule, extract, make_timesteps


class TestDiffusionSampling(unittest.TestCase):
    
    def test_cosine_beta_schedule(self):
        """Test the cosine_beta_schedule function."""
        timesteps = 10
        s = 0.008
        dtype = tf.float32

        betas = cosine_beta_schedule(timesteps, s, dtype)

        self.assertIsInstance(betas, tf.Tensor, "Output is not a tf.Tensor")

        self.assertEqual(betas.dtype, dtype, f"Expected dtype {dtype}, but got {betas.dtype}")

        self.assertEqual(betas.shape[0], timesteps, f"Expected length {timesteps}, but got {betas.shape[0]}")

        self.assertTrue(tf.reduce_all((betas >= 0) & (betas <= 0.999)), 
                        "Betas are out of expected range [0, 0.999]")

        # check betas and expected_betas_clipped are close
        alphas_cumprod = np.cos(((np.linspace(0, timesteps + 1, timesteps + 1) / (timesteps + 1)) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        expected_betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        expected_betas_clipped = np.clip(expected_betas, a_min=0, a_max=0.999)

        self.assertTrue(
            tf.reduce_all(tf.abs(betas - tf.convert_to_tensor(expected_betas_clipped, dtype=dtype)) < 1e-6),
            "Betas do not match the expected values"
        )

    def test_extract(self):
        """Test the extract function."""
        a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)  # Shape: (2, 3)
        t = tf.constant([0, 2], dtype=tf.int32)  # Shape: (2,)
        x_shape = (2, 1, 1)

        output = extract(a, t[:, None], x_shape)

        expected_output = tf.constant([[[1]], [[6]]], dtype=tf.float32)

        self.assertEqual(output.shape, (2, 1, 1), "Output shape is incorrect.")
        tf.debugging.assert_near(output, expected_output, message="Output values are incorrect.")

    def test_make_timesteps(self):
        """Test the make_timesteps function with a single set of parameters."""
        batch_size = 4
        i = 10
        
        t = make_timesteps(batch_size, i)
        
        self.assertIsInstance(t, tf.Tensor, "Output is not a tf.Tensor")
        self.assertEqual(t.dtype, tf.int64, f"Expected dtype tf.int64, but got {t.dtype}")
        self.assertEqual(t.shape, (batch_size,), f"Expected shape {(batch_size,)}, but got {t.shape}")
        self.assertTrue(tf.reduce_all(tf.equal(t, i)), 
                        f"All elements in the tensor should be {i}, but got {t.numpy()}")

    def test_make_timesteps_parametrized(self):
        """Test the make_timesteps function with multiple sets of parameters."""
        batch_size_list = [1, 5, 10]
        index_list = [0, 42, 7]

        for id in range(3):
            batch_size = batch_size_list[id]
            i = index_list[id]
            t = make_timesteps(batch_size, i)

            self.assertIsInstance(t, tf.Tensor, "Output is not a tf.Tensor")
            self.assertEqual(t.dtype, tf.int64, f"Expected dtype tf.int64, but got {t.dtype}")
            self.assertEqual(t.shape, (batch_size,), f"Expected shape {(batch_size,)}, but got {t.shape}")
            self.assertTrue(tf.reduce_all(tf.equal(t, i)), 
                           f"All elements in the tensor should be {i}, but got {t.numpy()}")


if __name__ == "__main__":
    unittest.main()
