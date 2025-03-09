import tensorflow as tf
import numpy as np
from tensorflow.keras.saving import register_keras_serializable


from util.torch_to_tf import nn_Linear, nn_ReLU, nn_LayerNorm


np.random.seed(42)  # set NumPy random seed
tf.random.set_seed(42)  # set TensorFlow random seed
import random
random.seed(42)

# sub-class C
@register_keras_serializable(package="Custom")
class C(tf.keras.Model):
    def __init__(self, units=4, **kwargs):
        super(C, self).__init__(**kwargs)
        self.units = units
        self.dense_c = nn_Linear(8, self.units)
        self.relu = nn_ReLU()

        self.layernorm = nn_LayerNorm(self.units)

    def call(self, inputs):
        x = self.dense_c(inputs)
        x = self.layernorm(x)

        return self.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# sub-class B
@register_keras_serializable(package="Custom")
class B(tf.keras.Model):
    def __init__(self, units=8, sub_model=None, **kwargs):
        super(B, self).__init__(**kwargs)
        self.units = units
        self.dense_b = nn_Linear(16, self.units)
        self.relu = nn_ReLU()
        self.c = sub_model if sub_model else C()

        self.layernorm = nn_LayerNorm(self.units)

    def call(self, inputs):
        x = self.dense_b(inputs)
        x = self.relu(x)
        x = self.layernorm(x)

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


# main class A
@register_keras_serializable(package="Custom")
class A(tf.keras.Model):
    def __init__(self, units=16, sub_model=None, **kwargs):
        super(A, self).__init__(**kwargs)
        self.units = units

        self.dense_a = nn_Linear(10, self.units)

        self.layernorm = nn_LayerNorm(self.units)

        self.relu = nn_ReLU()
        self.b = sub_model if sub_model else B()

    def call(self, inputs):
        x = self.dense_a(inputs)
        x = self.relu(x)
        x = self.layernorm(x)
        return self.b(x)

    def get_config(self):
        config = super().get_config()

        print("self.b = ", self.b)

        config.update({
            "units": self.units,
            "sub_model": tf.keras.layers.serialize(self.b),
        })
        return config

    @classmethod
    def from_config(cls, config):
        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"))
        return cls(sub_model=sub_model, **config)


# create model instance
model_a = A(units=16, sub_model=B(units=8, sub_model=C(units=4)))

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# data
x_train = tf.random.normal((32, 10))  # input shape: 32 samples, each with dimension 10
y_train = tf.random.normal((32, 4))   # input shape: 32 samples, each with dimension 4

# define loss function
mse_loss_fn = tf.keras.losses.MeanSquaredError()

# training procedures
for epoch in range(3):  # train 3 epochs
    print(f"Epoch {epoch + 1}")
    for step in range(1):  # iterate data (this is the simplified version)
        with tf.GradientTape() as tape:
            predictions = model_a(x_train)  # forward pass
            print(f"y_train shape: {y_train.shape}, predictions shape: {predictions.shape}")
            loss = mse_loss_fn(y_train, predictions)  # calculate loss

        gradients = tape.gradient(loss, model_a.trainable_variables)  # calculate gradients
        optimizer.apply_gradients(zip(gradients, model_a.trainable_variables))  # apply gradients

        print(f"Step {step + 1}, Loss: {loss.numpy():.4f}")

# save the model
model_a.save("nested_model.keras")

# load the model
loaded_model_a = tf.keras.models.load_model("nested_model.keras", custom_objects={"A": A, "B": B, "C": C})

# check if the weights are the same
outputs_original = model_a(x_train)
loss1 = mse_loss_fn(y_train, outputs_original)  # calculate the loss

outputs_loaded = loaded_model_a(x_train)
loss2 = mse_loss_fn(y_train, outputs_loaded)  # calculate the loss

print(f"Loss1: {loss1.numpy():.4f}")
print(f"Loss2: {loss2.numpy():.4f}")


assert np.allclose(outputs_original.numpy(), outputs_loaded.numpy())
print("Model outputs are consistent after saving and loading.")


