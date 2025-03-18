import unittest
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.saving import register_keras_serializable

from util.torch_to_tf import nn_Linear, nn_ReLU, nn_LayerNorm

from tensorflow.keras.utils import get_custom_objects


# sub-class C
@register_keras_serializable(package="Custom")
class C8(tf.keras.Model):
    def __init__(self, units=4, dense_c=None, **kwargs):
        super(C8, self).__init__(**kwargs)
        self.units = units
        if not dense_c:
            self.dense_c = nn_Linear(8, self.units)
        else:
            self.dense_c = dense_c
        self.relu = nn_ReLU()
        self.layernorm = nn_LayerNorm(self.units)

    def call(self, inputs):
        x = self.dense_c(inputs)
        x = self.layernorm(x)

        return self.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dense_c": tf.keras.layers.serialize(self.dense_c),
        })

        print("C.get_config(): config = ", config)

        return config

    @classmethod
    def from_config(cls, config):

        cur_dict = {
            'nn_Linear': nn_Linear,
            'nn_ReLU': nn_ReLU,
            "nn_LayerNorm": nn_LayerNorm,
        }
        get_custom_objects().update(cur_dict)
        
        print("C.from_config(): config = ", config)

        if "dense_c" in config:
            print("'dense_c' in config")
        else:
            print("'dense_c' not in config")

        dense_c = tf.keras.layers.deserialize(config.pop("dense_c"),  custom_objects=get_custom_objects())
        
        return cls(dense_c = dense_c, **config)


# sub-class B
@register_keras_serializable(package="Custom")
class B8(tf.keras.Model):
    def __init__(self, units=8, sub_model=None, dense_b = None, **kwargs):
        super(B8, self).__init__(**kwargs)
        self.units = units

        if not dense_b:
            self.dense_b = nn_Linear(16, self.units)
        else:
            self.dense_b = dense_b

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
            "dense_b": tf.keras.layers.serialize(self.dense_b),
        })
        print("B.get_config(): config = ", config)
        return config

    @classmethod
    def from_config(cls, config):

        from tensorflow.keras.utils import get_custom_objects

        cur_dict = {
            # 'A8': A8,
            'B': B8,
            'C': C8,  
            'nn_Linear': nn_Linear,
            'nn_ReLU': nn_ReLU,
            "nn_LayerNorm": nn_LayerNorm,
        }
        get_custom_objects().update(cur_dict)

        print("B.from_config(): config = ", config)


        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"),  custom_objects=get_custom_objects())
        dense_b = tf.keras.layers.deserialize(config.pop("dense_b"),  custom_objects=get_custom_objects())


        return cls(sub_model=sub_model, dense_b=dense_b, **config)


# main class A
@register_keras_serializable(package="Custom")
class A8(tf.keras.Model):
    def __init__(self, units=16, sub_model=None, dense_a = None, **kwargs):
        super(A8, self).__init__(**kwargs)
        self.units = units

        if not dense_a:
            self.dense_a = nn_Linear(10, self.units)
        else:
            self.dense_a = dense_a
            
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

        # print("self.b = ", self.b)

        cur_dict = {
            # 'A8': A8,
            'B8': B8,
            'C8': C8,  
            'nn_Linear': nn_Linear,
            'nn_ReLU': nn_ReLU,
            "nn_LayerNorm": nn_LayerNorm,
        }
        get_custom_objects().update(cur_dict)


        config.update({
            "units": self.units,
            "sub_model": tf.keras.layers.serialize(self.b),
            "dense_a": tf.keras.layers.serialize(self.dense_a),
        })

        print("A.get_config(): config = ", config)

        return config

    @classmethod
    def from_config(cls, config):
        from tensorflow.keras.utils import get_custom_objects
        cur_dict = {
            # 'A8': A8,
            'B8': B8,
            'C8': C8,  
            'nn_Linear': nn_Linear,
            'nn_ReLU': nn_ReLU,
            "nn_LayerNorm": nn_LayerNorm,
        }
        get_custom_objects().update(cur_dict)

        print("A.from_config(): config = ", config)

        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"),  custom_objects=get_custom_objects())
        dense_a = tf.keras.layers.deserialize(config.pop("dense_a"),  custom_objects=get_custom_objects())
        return cls(sub_model=sub_model, dense_a=dense_a, **config)


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
        self.model_path = "nested_model8.keras"

    def tearDown(self):
        """Clean up after each test method."""
        # Remove saved model file if it exists
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_model_creation(self):
        """Test that nested models can be created properly."""
        # Create the model instance with nested structure
        model_a = A8(units=16, sub_model=B8(units=8, sub_model=C8(units=4)))
        
        # Test forward pass
        output = model_a(self.x_train)
        
        # Check output shape
        self.assertEqual(output.shape, (32, 4))

    def test_save_load_consistency(self):
        """Test that model outputs are consistent after saving and loading."""
        # Create and train the model
        model_a = A8(units=16, sub_model=B8(units=8, sub_model=C8(units=4)))
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        

        # Training procedures
        for epoch in range(3):  # train 3 epochs
            for step in range(1):  # iterate data (simplified version)
                with tf.GradientTape() as tape:
                    predictions = model_a(self.x_train)  # forward pass
                    loss = self.mse_loss_fn(self.y_train, predictions)  # calculate loss
                
                gradients = tape.gradient(loss, model_a.trainable_variables)  # calculate gradients
                optimizer.apply_gradients(zip(gradients, model_a.trainable_variables))  # apply gradients
        
        outputs_original = model_a(self.x_train)
        loss_original = self.mse_loss_fn(self.y_train, outputs_original)




        # Save the model
        model_a.save(self.model_path)
        
        # Verify the model file exists
        self.assertTrue(os.path.exists(self.model_path))
        
        # Load the model
        loaded_model_a = tf.keras.models.load_model(
            self.model_path, 
            custom_objects = {"A8": A8, "B8": B8, "C8": C8,
                  "nn_Linear": nn_Linear, "nn_ReLU": nn_ReLU, "nn_LayerNorm": nn_LayerNorm}
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
