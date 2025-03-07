import tensorflow as tf
import tensorflow_probability as tfp

# define mean and covariance
mean = tf.constant([0.0, 0.0])  # mean vector (D=2)
covariance_matrix = tf.constant([[1.0, 0.5], [0.5, 1.0]])  # covariance matrix (D=2, D=2)

# create multi-variates normal distribution
# distribution = tfp.distributions.MultivariateNormalFullCovariance(
#     loc=mean,
#     covariance_matrix=covariance_matrix
# )

distribution = tfp.distributions.MultivariateNormalTriL(
    loc=mean,
    scale_tril=tf.linalg.cholesky(covariance_matrix)
)

# calculate the probability densitysss
x = tf.constant([1.0, 1.0])  # the input point
pdf_value = distribution.prob(x)  # calculate the value of pdf
print(pdf_value.numpy())

