import unittest
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.saving import register_keras_serializable

from util.torch_to_tf import nn_Linear, nn_ReLU, nn_LayerNorm


# Sub-class C
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


# Sub-class B
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


# Main class A
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


class TestNestedModelSaveLoad(unittest.TestCase):
    """Test case for saving and loading nested TensorFlow models."""
    
    def setUp(self):
        """Set up test data and model for each test."""
        # Create model instance with nested structure
        self.model_a = A(units=16, sub_model=B(units=8, sub_model=C(units=4)))
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Create test data
        self.x_train = tf.random.normal((32, 10))  # input shape: 32 samples, each with dimension 10
        self.y_train = tf.random.normal((32, 4))   # output shape: 32 samples, each with dimension 4
        
        # Define loss function
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Model save path
        self.model_path = "nested_model.keras"

    def tearDown(self):
        """Clean up after each test."""
        # Remove saved model file if it exists
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_model_creation(self):
        """Test that the nested model can be created properly."""
        # Verify model structure
        self.assertIsInstance(self.model_a, A)
        self.assertIsInstance(self.model_a.b, B)
        self.assertIsInstance(self.model_a.b.c, C)
        
        # Verify model output shape
        output = self.model_a(self.x_train)
        self.assertEqual(output.shape, self.y_train.shape)

    def test_model_training(self):
        """Test that the model can be trained."""
        initial_output = self.model_a(self.x_train)
        initial_loss = self.mse_loss_fn(self.y_train, initial_output)
        
        # Train for 3 epochs
        for epoch in range(3):
            with tf.GradientTape() as tape:
                predictions = self.model_a(self.x_train)  # forward pass
                loss = self.mse_loss_fn(self.y_train, predictions)  # calculate loss using MSE
            
            gradients = tape.gradient(loss, self.model_a.trainable_variables)  # calculate gradients
            self.optimizer.apply_gradients(zip(gradients, self.model_a.trainable_variables))  # apply gradients
        
        # Verify that loss decreased after training
        final_output = self.model_a(self.x_train)
        final_loss = self.mse_loss_fn(self.y_train, final_output)
        
        # Loss should typically decrease after training
        self.assertLessEqual(final_loss, initial_loss)

    def test_save_and_load(self):
        """Test that the model can be saved and loaded with consistent results."""
        # Train the model for a single step to update weights
        with tf.GradientTape() as tape:
            predictions = self.model_a(self.x_train)
            loss = self.mse_loss_fn(self.y_train, predictions)
        
        gradients = tape.gradient(loss, self.model_a.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_a.trainable_variables))
        
        # Get outputs from original model
        outputs_original = self.model_a(self.x_train)
        
        # Save the model
        self.model_a.save(self.model_path)
        
        # Verify model file exists
        self.assertTrue(os.path.exists(self.model_path))
        
        # Load the model
        loaded_model_a = tf.keras.models.load_model(
            self.model_path, 
            custom_objects={"A": A, "B": B, "C": C}
        )
        
        # Get outputs from loaded model
        outputs_loaded = loaded_model_a(self.x_train)
        
        # Calculate losses for comparison
        loss1 = self.mse_loss_fn(self.y_train, outputs_original)
        loss2 = self.mse_loss_fn(self.y_train, outputs_loaded)
        
        # Verify losses are the same
        self.assertAlmostEqual(loss1.numpy(), loss2.numpy(), places=5)
        
        # Verify outputs are the same
        self.assertTrue(np.allclose(outputs_original.numpy(), outputs_loaded.numpy()))


if __name__ == '__main__':
    unittest.main()
