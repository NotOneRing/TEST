import tensorflow as tf

import numpy as np

from util.torch_to_tf import nn_Linear, nn_ReLU

from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
class C(tf.keras.Model):
    def __init__(self, **kwargs):
        super(C, self).__init__()
        self.dense_c = nn_Linear(8, 4)
        # tf.keras.layers.Dense(4, activation='relu')
        self.relu = nn_ReLU()
    
    def call(self, inputs):
        x = self.dense_c(inputs)
        return self.relu(x)


@register_keras_serializable(package="Custom")
class B(tf.keras.Model):
    def __init__(self, **kwargs):
        super(B, self).__init__()
        self.dense_b = nn_Linear(16, 8)
        # tf.keras.layers.Dense(8, activation='relu')
        self.relu = nn_ReLU()
        self.c = C()
    
    def call(self, inputs):
        x = self.dense_b(inputs)
        x = self.relu(x)
        return self.c(x)


@register_keras_serializable(package="Custom")
class A(tf.keras.Model):
    def __init__(self, **kwargs):
        super(A, self).__init__()
        self.dense_a = nn_Linear(10, 16)
        # tf.keras.layers.Dense(16, activation='relu')
        self.relu = nn_ReLU()
        self.b = B()
    
    def call(self, inputs):
        x = self.dense_a(inputs)
        x = self.relu(x)
        return self.b(x)





# 创建模型实例
model_a = A()

# 编译和训练
model_a.compile(optimizer='adam', loss='mse')
x_train = tf.random.normal((32, 10))  # 输入形状：32个样本，每个样本10维
y_train = tf.random.normal((32, 4))   # 输出形状：32个样本，每个样本4维
model_a.fit(x_train, y_train, epochs=3)






# 保存模型（会递归保存B和C的所有内容）
model_a.save("nested_model.keras")





# 加载模型
loaded_model_a = tf.keras.models.load_model("nested_model.keras")



# 检查是否保留了权重
outputs_original = model_a(x_train)
outputs_loaded = loaded_model_a(x_train)

print("outputs_original = ", outputs_original)
print("outputs_loaded = ", outputs_loaded)

assert np.allclose(outputs_original.numpy(), outputs_loaded.numpy())

print(tf.reduce_sum(tf.abs(outputs_original - outputs_loaded)))  # 应接近于0

# 检查优化器状态
print(loaded_model_a.optimizer.get_config())













