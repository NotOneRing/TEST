import unittest
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.saving import register_keras_serializable

from util.torch_to_tf import nn_Linear, nn_ReLU, nn_LayerNorm


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

        self.network = nn_Linear(10, self.units)

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
    """Test case for saving and loading nested Keras models."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        import random
        random.seed(42)
        
        # Define test data
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

    def test_model_creation(self):
        """Test that nested models can be created properly."""
        # Create the model instance with nested structure
        model_a = A(units=16, sub_model=B(units=8, sub_model=C(units=4)))
        
        # Test forward pass
        output = model_a(self.x_train)
        
        # Check output shape
        self.assertEqual(output.shape, (32, 4))

    def test_model_training(self):
        """Test that the model can be trained."""
        # Create the model instance
        model_a = A(units=16, sub_model=B(units=8, sub_model=C(units=4)))
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Initial prediction
        initial_pred = model_a(self.x_train)
        initial_loss = self.mse_loss_fn(self.y_train, initial_pred)
        
        # Training procedures
        for epoch in range(3):  # train 3 epochs
            for step in range(1):  # iterate data (simplified version)
                with tf.GradientTape() as tape:
                    predictions = model_a(self.x_train)  # forward pass
                    loss = self.mse_loss_fn(self.y_train, predictions)  # calculate loss
                
                gradients = tape.gradient(loss, model_a.trainable_variables)  # calculate gradients
                optimizer.apply_gradients(zip(gradients, model_a.trainable_variables))  # apply gradients
        
        # Final prediction
        final_pred = model_a(self.x_train)
        final_loss = self.mse_loss_fn(self.y_train, final_pred)
        
        # Check that loss decreased after training
        self.assertLess(final_loss, initial_loss)

    def test_save_load_consistency(self):
        """Test that model outputs are consistent after saving and loading."""
        # Create and train the model
        model_a = A(units=16, sub_model=B(units=8, sub_model=C(units=4)))
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Training procedures (simplified)
        for epoch in range(3):
            with tf.GradientTape() as tape:
                predictions = model_a(self.x_train)
                loss = self.mse_loss_fn(self.y_train, predictions)
            
            gradients = tape.gradient(loss, model_a.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_a.trainable_variables))
        
        # Get outputs from original model
        outputs_original = model_a(self.x_train)
        loss_original = self.mse_loss_fn(self.y_train, outputs_original)
        
        # Save the model
        model_a.save(self.model_path)
        
        # Verify the model file exists
        self.assertTrue(os.path.exists(self.model_path))
        
        # Load the model
        loaded_model_a = tf.keras.models.load_model(
            self.model_path, 
            custom_objects={"A": A, "B": B, "C": C}
        )
        
        # Get outputs from loaded model
        outputs_loaded = loaded_model_a(self.x_train)
        loss_loaded = self.mse_loss_fn(self.y_train, outputs_loaded)
        
        # Check that losses are the same
        self.assertAlmostEqual(loss_original.numpy(), loss_loaded.numpy(), places=5)
        
        # Check that outputs are the same
        self.assertTrue(np.allclose(outputs_original.numpy(), outputs_loaded.numpy()))

        self.assertTrue( np.allclose(model_a.dense_a.kernel.numpy(), loaded_model_a.dense_a.kernel.numpy()) )



if __name__ == "__main__":
    unittest.main()
