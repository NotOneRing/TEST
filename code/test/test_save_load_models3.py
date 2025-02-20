import tensorflow as tf
import numpy as np
from util.torch_to_tf import nn_Linear, nn_ReLU
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
class C(tf.keras.Model):
    def __init__(self, units=4, **kwargs):
        super(C, self).__init__(**kwargs)
        self.units = units
        self.dense_c = nn_Linear(8, self.units)
        self.relu = nn_ReLU()

    def call(self, inputs):
        x = self.dense_c(inputs)
        return self.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="Custom")
class B(tf.keras.Model):
    def __init__(self, units=8, sub_model=None, **kwargs):
        super(B, self).__init__(**kwargs)
        self.units = units
        self.dense_b = nn_Linear(16, self.units)
        self.relu = nn_ReLU()
        self.c = sub_model if sub_model else C()

    def call(self, inputs):
        x = self.dense_b(inputs)
        x = self.relu(x)
        return self.c(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sub_model": tf.keras.layers.serialize(self.c),
        })
        return config

    @classmethod
    def from_config(cls, config):
        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"))
        return cls(sub_model=sub_model, **config)


@register_keras_serializable(package="Custom")
class A(tf.keras.Model):
    def __init__(self, units=16, sub_model=None, **kwargs):
        super(A, self).__init__(**kwargs)
        self.units = units
        self.dense_a = nn_Linear(10, self.units)
        self.relu = nn_ReLU()
        self.b = sub_model if sub_model else B()

    def call(self, inputs):
        x = self.dense_a(inputs)
        x = self.relu(x)
        return self.b(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sub_model": tf.keras.layers.serialize(self.b),
        })
        return config

    @classmethod
    def from_config(cls, config):
        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"))
        return cls(sub_model=sub_model, **config)


# 创建模型实例
model_a = A()

# 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 数据
x_train = tf.random.normal((32, 10))  # 输入形状：32个样本，每个样本10维
y_train = tf.random.normal((32, 4))   # 输出形状：32个样本，每个样本4维

# 定义损失函数
mse_loss_fn = tf.keras.losses.MeanSquaredError()

# 训练步骤
for epoch in range(3):  # 训练 3 个 epoch
    print(f"Epoch {epoch + 1}")
    for step in range(1):  # 遍历数据 (这里是简化版本)
        with tf.GradientTape() as tape:
            predictions = model_a(x_train)  # 前向传播
            loss = mse_loss_fn(y_train, predictions)  # 计算损失

        gradients = tape.gradient(loss, model_a.trainable_variables)  # 计算梯度
        optimizer.apply_gradients(zip(gradients, model_a.trainable_variables))  # 应用梯度

        print(f"Step {step + 1}, Loss: {loss.numpy():.4f}")

# 保存模型
model_a.save("nested_model.keras")

# 加载模型
loaded_model_a = tf.keras.models.load_model("nested_model.keras")

# 检查是否保留了权重
outputs_original = model_a(x_train)
outputs_loaded = loaded_model_a(x_train)

assert np.allclose(outputs_original.numpy(), outputs_loaded.numpy())


print(tf.reduce_sum(tf.abs(outputs_original - outputs_loaded)))  # 应接近于0

