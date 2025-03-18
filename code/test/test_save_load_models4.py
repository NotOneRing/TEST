import tensorflow as tf
import numpy as np
import unittest
import os
from tensorflow.keras.saving import register_keras_serializable


# Custom linear layer
@register_keras_serializable(package="Custom")
class nn_Linear(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(nn_Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = self.add_weight(
            shape=(input_dim, output_dim),
            initializer="random_normal",
            trainable=True,
            name="weights"
        )
        self.b = self.add_weight(
            shape=(output_dim,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config


# Custom activation
@register_keras_serializable(package="Custom")
class nn_ReLU(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.relu(inputs)


# Sub-class model C
@register_keras_serializable(package="Custom")
class C4(tf.keras.Model):
    def __init__(self, units=4, **kwargs):
        super(C4, self).__init__(**kwargs)
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


# Sub-class model B
@register_keras_serializable(package="Custom")
class B4(tf.keras.Model):
    def __init__(self, units=8, sub_model=None, **kwargs):
        super(B4, self).__init__(**kwargs)
        self.units = units
        self.dense_b = nn_Linear(16, self.units)
        self.relu = nn_ReLU()
        self.c = sub_model if sub_model else C4()

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


# Main model A
@register_keras_serializable(package="Custom")
class A4(tf.keras.Model):
    def __init__(self, units=16, sub_model=None, **kwargs):
        super(A4, self).__init__(**kwargs)
        self.units = units
        self.dense_a = nn_Linear(10, self.units)
        self.relu = nn_ReLU()
        self.b = sub_model if sub_model else B4()

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


class TestNestedModelSaveLoad(unittest.TestCase):
    """Test case for saving and loading nested TensorFlow models."""
    
    def setUp(self):
        """Set up test data and model."""
        # Create model instance with nested structure
        self.model_a = A4(units=16, sub_model=B4(units=8, sub_model=C4(units=4)))
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Test data
        self.x_train = tf.random.normal((32, 10))  # input shape: 32 samples, each with dimension 10
        self.y_train = tf.random.normal((32, 4))   # output shape: 32 samples, each with dimension 4
        
        # Define loss function
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Model save path
        self.model_path = "nested_model4.keras"
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove saved model file if it exists
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
    
    def test_model_training(self):
        """Test that the model can be trained."""
        # Training procedures
        for epoch in range(3):  # train 3 epochs
            for step in range(1):  # iterate data (simplified version)
                with tf.GradientTape() as tape:
                    predictions = self.model_a(self.x_train)  # forward pass
                    loss = self.mse_loss_fn(self.y_train, predictions)  # calculate loss
                
                # Calculate and apply gradients
                gradients = tape.gradient(loss, self.model_a.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model_a.trainable_variables))
                
                # Verify loss is a valid number
                self.assertFalse(np.isnan(loss.numpy()))
    
        # Save the model
        self.model_a.save(self.model_path)
        
        # Load the model
        loaded_model_a = tf.keras.models.load_model(
            self.model_path, 
            custom_objects={"A4": A4, "B4": B4, "C4": C4, "nn_Linear": nn_Linear, "nn_ReLU": nn_ReLU}
        )
        
        # Get outputs from both models
        outputs_original = self.model_a(self.x_train)
        outputs_loaded = loaded_model_a(self.x_train)
        
        # Calculate losses for comparison
        loss1 = self.mse_loss_fn(self.y_train, outputs_original)
        loss2 = self.mse_loss_fn(self.y_train, outputs_loaded)
        
        # Verify losses are the same
        self.assertAlmostEqual(loss1.numpy(), loss2.numpy(), places=5)
        
        # Verify outputs are the same
        self.assertTrue(np.allclose(outputs_original.numpy(), outputs_loaded.numpy()))


if __name__ == "__main__":
    unittest.main()
