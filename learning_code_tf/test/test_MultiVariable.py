import tensorflow as tf
import tensorflow_probability as tfp

# 定义均值和协方差矩阵
mean = tf.constant([0.0, 0.0])  # 均值向量 (D=2)
covariance_matrix = tf.constant([[1.0, 0.5], [0.5, 1.0]])  # 协方差矩阵 (D=2, D=2)

# 创建多元正态分布
# distribution = tfp.distributions.MultivariateNormalFullCovariance(
#     loc=mean,
#     covariance_matrix=covariance_matrix
# )

distribution = tfp.distributions.MultivariateNormalTriL(
    loc=mean,
    scale_tril=tf.linalg.cholesky(covariance_matrix)
)

# 计算概率密度
x = tf.constant([1.0, 1.0])  # 输入点
pdf_value = distribution.prob(x)  # 计算 pdf 值
print(pdf_value.numpy())

