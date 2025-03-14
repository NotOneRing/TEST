import unittest
import tensorflow as tf
import numpy as np
import os
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


class TestNestedModelSaveLoad(unittest.TestCase):
    """Test case for saving and loading nested TensorFlow models."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create model instance
        self.model_a = A()
        
        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Create test data
        self.x_train = tf.random.normal((32, 10))  # input shape: 32 samples, each with dimension 10
        self.y_train = tf.random.normal((32, 4))   # output shape: 32 samples, each with dimension 4
        
        # Define loss function
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Model save path
        self.model_path = "nested_model.keras"
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove saved model file if it exists
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
    
    def test_model_training(self):
        """Test that the model can be trained successfully."""
        # Training procedures
        for epoch in range(3):  # train 3 epochs
            print(f"Epoch {epoch + 1}")
            for step in range(1):  # iterate data (this is the simplified version)
                with tf.GradientTape() as tape:
                    predictions = self.model_a(self.x_train)  # forward pass
                    loss = self.mse_loss_fn(self.y_train, predictions)  # calculate loss
                
                # Calculate gradient
                gradients = tape.gradient(loss, self.model_a.trainable_variables)
                # Apply gradient
                self.optimizer.apply_gradients(zip(gradients, self.model_a.trainable_variables))
                
                # print(f"Step {step + 1}, Loss: {loss.numpy():.4f}")
                
                # Verify loss is a valid number
                self.assertFalse(np.isnan(loss.numpy()), "Loss should not be NaN")
    
    def test_save_load_model(self):
        """Test saving and loading the model."""
        # First train the model
        for epoch in range(3):
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
        self.assertTrue(os.path.exists(self.model_path), "Model file should exist after saving")
        
        # Load the model
        loaded_model_a = tf.keras.models.load_model(self.model_path)
        
        # Get outputs from loaded model
        outputs_loaded = loaded_model_a(self.x_train)
        
        # Check if outputs are the same
        self.assertTrue(
            np.allclose(outputs_original.numpy(), outputs_loaded.numpy()),
            "Outputs from original and loaded models should be the same"
        )
        
        # Calculate absolute difference
        diff = tf.reduce_sum(tf.abs(outputs_original - outputs_loaded))
        print(f"Sum of absolute differences: {diff.numpy()}")
        
        # Verify difference is close to zero
        self.assertLess(diff.numpy(), 1e-5, "Difference between outputs should be close to zero")
    
    def test_model_architecture(self):
        """Test that the model architecture is preserved after loading."""
        # Save the model
        self.model_a.save(self.model_path)
        
        # Load the model
        loaded_model_a = tf.keras.models.load_model(self.model_path)
        
        # Check model structure
        self.assertIsInstance(loaded_model_a, A, "Loaded model should be an instance of A")
        self.assertIsInstance(loaded_model_a.b, B, "Loaded model's b attribute should be an instance of B")
        self.assertIsInstance(loaded_model_a.b.c, C, "Loaded model's b.c attribute should be an instance of C")
        
        # Check model parameters
        self.assertEqual(loaded_model_a.units, 16, "Model A should have 16 units")
        self.assertEqual(loaded_model_a.b.units, 8, "Model B should have 8 units")
        self.assertEqual(loaded_model_a.b.c.units, 4, "Model C should have 4 units")


if __name__ == "__main__":
    unittest.main()
