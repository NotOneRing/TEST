import unittest
import tensorflow as tf
import tensorflow_probability as tfp


class TestMultivariateNormal(unittest.TestCase):
    def setUp(self):
        # Define mean and covariance
        self.mean = tf.constant([0.0, 0.0])  # mean vector (D=2)
        self.covariance_matrix = tf.constant([[1.0, 0.5], [0.5, 1.0]])  # covariance matrix (D=2, D=2)
        
        self.distribution = tfp.distributions.MultivariateNormalTriL(
            loc=self.mean,
            scale_tril=tf.linalg.cholesky(self.covariance_matrix)
        )
        
        self.x = tf.constant([1.0, 1.0])

    def test_distribution_creation(self):
        """Test that the distribution is created correctly"""
        self.assertEqual(self.distribution.event_shape, tf.TensorShape([2]))
        self.assertEqual(self.distribution.batch_shape, tf.TensorShape([]))


    def test_pdf_calculation(self):
        """Test the probability density function calculation"""
        pdf_value = self.distribution.prob(self.x)
        
        self.assertGreater(pdf_value.numpy(), 0)
        
        # We can also test the specific value if we know what it should be
        # This is based on the original code's output
        expected_pdf = 0.09435

        # print("pdf_value.numpy() = ", pdf_value.numpy())
        self.assertAlmostEqual(pdf_value.numpy(), expected_pdf, places=5)


    def test_log_prob(self):
        """Test the log probability calculation"""
        log_pdf = self.distribution.log_prob(self.x)
        pdf = self.distribution.prob(self.x)
        
        # The log_prob should be the natural logarithm of prob
        expected_log_pdf = tf.math.log(pdf)
        self.assertAlmostEqual(log_pdf.numpy(), expected_log_pdf.numpy(), places=5)


if __name__ == '__main__':
    unittest.main()
