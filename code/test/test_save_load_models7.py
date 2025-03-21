import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras.saving import register_keras_serializable

import os

from util.torch_to_tf import nn_Linear, nn_ReLU, nn_LayerNorm



# sub-class C
@register_keras_serializable(package="Custom")
class C7(tf.keras.Model):
    def __init__(self, units=4, **kwargs):
        super(C7, self).__init__(**kwargs)
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
class B7(tf.keras.Model):
    def __init__(self, units=8, sub_model=None, **kwargs):
        super(B7, self).__init__(**kwargs)
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

        from tensorflow.keras.utils import get_custom_objects
        cur_dict = {
            'A7': A7,
            'B7': B7,
            'C7': C7,  
            'nn_Linear': nn_Linear,
            'nn_ReLU': nn_ReLU,
            "nn_LayerNorm": nn_LayerNorm,
        }
        get_custom_objects().update(cur_dict)
        
        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"),  custom_objects=get_custom_objects())
        return cls(sub_model=sub_model, **config)

# main class A
@register_keras_serializable(package="Custom")
class A7(tf.keras.Model):
    def __init__(self, units=16, sub_model=None, **kwargs):
        super(A7, self).__init__(**kwargs)
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
        # print("self.b = ", self.b)
        config.update({
            "units": self.units,
            "sub_model": tf.keras.layers.serialize(self.b),
        })
        return config

    @classmethod
    def from_config(cls, config):
        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"))
        return cls(sub_model=sub_model, **config)



class TestSaveLoadNestedModels(unittest.TestCase):
    """Test case for saving and loading nested TensorFlow models."""
    
    def setUp(self):
        """Set up test environment with fixed random seeds."""
        np.random.seed(42)  # set NumPy random seed
        tf.random.set_seed(42)  # set TensorFlow random seed
        import random
        random.seed(42)
        
        # Define test data
        self.x_train = tf.random.normal((32, 10))  # input shape: 32 samples, each with dimension 10
        self.y_train = tf.random.normal((32, 4))   # input shape: 32 samples, each with dimension 4
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Model save path
        self.model_path = "nested_model7.keras"

    def tearDown(self):
        """Clean up after each test."""
        # Remove saved model file if it exists
        if os.path.exists(self.model_path):
            os.remove(self.model_path)


    def test_model_training(self):
        """Test that the nested model can be trained properly."""
        # Create model instance
        model_a = A7(units=16, sub_model=B7(units=8, sub_model=C7(units=4)))
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Training procedures
        for epoch in range(3):  # train 3 epochs
            # print(f"Epoch {epoch + 1}")
            for step in range(1):  # iterate data (this is the simplified version)
                with tf.GradientTape() as tape:
                    predictions = model_a(self.x_train)  # forward pass
                    # print(f"y_train shape: {self.y_train.shape}, predictions shape: {predictions.shape}")
                    loss = self.mse_loss_fn(self.y_train, predictions)  # calculate loss

                gradients = tape.gradient(loss, model_a.trainable_variables)  # calculate gradients
                optimizer.apply_gradients(zip(gradients, model_a.trainable_variables))  # apply gradients

                # print(f"Step {step + 1}, Loss: {loss.numpy():.4f}")
        
        # Verify model produces output of expected shape
        predictions = model_a(self.x_train)
        self.assertEqual(predictions.shape, self.y_train.shape)
        
        return model_a  # Return the trained model for use in other tests
    
    def test_save_load_model(self):
        """Test that the model can be saved and loaded with consistent outputs."""
        # Train the model
        model_a = self.test_model_training()
        
        # Save the model
        model_a.save(self.model_path)
        
        # Load the model
        loaded_model_a = tf.keras.models.load_model(
            self.model_path, 
            # custom_objects={
            #     "A7": self.A7, 
            #     "B7": self.B7, 
            #     "C7": self.C7
            # }
            custom_objects = {"A7": A7, "B7": B7, "C7": C7,
                  "nn_Linear": nn_Linear, "nn_ReLU": nn_ReLU, "nn_LayerNorm": nn_LayerNorm}

        )
        
        # Check if the weights are the same by comparing outputs
        outputs_original = model_a(self.x_train)
        loss1 = self.mse_loss_fn(self.y_train, outputs_original)
        
        outputs_loaded = loaded_model_a(self.x_train)
        loss2 = self.mse_loss_fn(self.y_train, outputs_loaded)
        
        # print(f"Loss1: {loss1.numpy():.4f}")
        # print(f"Loss2: {loss2.numpy():.4f}")
        
        # Assert outputs are the same
        self.assertTrue(np.allclose(outputs_original.numpy(), outputs_loaded.numpy()))
        # print("Model outputs are consistent after saving and loading.")


if __name__ == "__main__":
    unittest.main()
