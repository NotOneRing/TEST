import unittest
import os
import tensorflow as tf
import numpy as np
from util.torch_to_tf import nn_Linear, nn_ReLU
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
class C2(tf.keras.Model):
    def __init__(self, units=4, dense_c = None, **kwargs):
        super(C2, self).__init__(**kwargs)
        self.units = units
        if dense_c:
            self.dense_c = dense_c
        else:
            self.dense_c = nn_Linear(8, self.units)
        self.relu = nn_ReLU()
    
    def call(self, inputs):
        x = self.dense_c(inputs)
        return self.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dense_c": tf.keras.layers.serialize(self.dense_c),
        })
        return config

    @classmethod
    def from_config(cls, config):
        dense_c = tf.keras.layers.deserialize(config.pop("dense_c"))
        return cls(dense_c = dense_c, **config)


@register_keras_serializable(package="Custom")
class B2(tf.keras.Model):
    def __init__(self, units=8, sub_model=None, dense_b = None, **kwargs):
        super(B2, self).__init__(**kwargs)
        self.units = units
        if not dense_b:
            self.dense_b = nn_Linear(16, self.units)
        else:
            self.dense_b = dense_b

        self.relu = nn_ReLU()
        self.c = sub_model if sub_model else C2()

    def call(self, inputs):
        x = self.dense_b(inputs)
        x = self.relu(x)
        return self.c(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sub_model": tf.keras.layers.serialize(self.c),
            "dense_b": tf.keras.layers.serialize(self.dense_b),
        })
        return config

    @classmethod
    def from_config(cls, config):
        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"))
        dense_b = tf.keras.layers.deserialize(config.pop("dense_b"))
        return cls(sub_model=sub_model, dense_b=dense_b, **config)


@register_keras_serializable(package="Custom")
class A2(tf.keras.Model):
    def __init__(self, units=16, sub_model=None, dense_a = None, **kwargs):
        super(A2, self).__init__(**kwargs)
        self.units = units
        if not dense_a:
            self.dense_a = nn_Linear(10, self.units)            
        else:
            self.dense_a = dense_a
        self.relu = nn_ReLU()
        self.b = sub_model if sub_model else B2()

    def call(self, inputs):
        x = self.dense_a(inputs)
        x = self.relu(x)
        return self.b(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sub_model": tf.keras.layers.serialize(self.b),
            "dense_a": tf.keras.layers.serialize(self.dense_a),
        })
        return config

    @classmethod
    def from_config(cls, config):
        sub_model = tf.keras.layers.deserialize(config.pop("sub_model"))
        dense_a = tf.keras.layers.deserialize(config.pop("dense_a"))
        return cls(sub_model=sub_model, dense_a=dense_a, **config)


class TestNestedModelSaveLoad(unittest.TestCase):
    """Test saving and loading of nested TensorFlow models"""
    
    def setUp(self):
        """Setup work before each test"""
        # Set model save path
        self.model_path = "nested_model2.keras"
        
        # Ensure the file doesn't exist before testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            
        # Create training data
        self.x_train = tf.random.normal((32, 10))  # Shape: 32 samples, 10 dimensions each
        self.y_train = tf.random.normal((32, 4))   # Shape: 32 samples, 4 dimensions each
    
    def tearDown(self):
        """Cleanup work after each test"""
        # Delete model file created during testing
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
    
    def test_nested_model_save_load(self):
        """Test saving and loading functionality of nested models"""
        # Create nested model
        model_a = A2()
        
        # Compile and train the model
        model_a.compile(optimizer='adam', loss='mse')
        model_a.fit(self.x_train, self.y_train, epochs=3, verbose=0)
        
        # Save the model
        model_a.save(self.model_path)
        
        # Verify model file was created
        self.assertTrue(os.path.exists(self.model_path), "Model file was not created successfully")

        from tensorflow.keras.utils import get_custom_objects
        cur_dict = {
            'A2': A2,
            'B2': B2,
            'C2': C2,  
            'nn_Linear': nn_Linear,
            'nn_ReLU': nn_ReLU,
        }
        get_custom_objects().update(cur_dict)
        loaded_model_a = tf.keras.models.load_model(self.model_path, custom_objects=get_custom_objects())
        
        # Check if outputs from original and loaded models are the same
        outputs_original = model_a(self.x_train)
        outputs_loaded = loaded_model_a(self.x_train)
        
        # # Print outputs for comparison
        # print("outputs_original = ", outputs_original)
        # print("outputs_loaded = ", outputs_loaded)
        
        # Verify outputs are close using numpy's allclose function
        self.assertTrue(
            np.allclose(outputs_original.numpy(), outputs_loaded.numpy()),
            "Outputs from original and loaded models do not match"
        )
        
        # Calculate sum of absolute differences, should be close to 0
        diff_sum = tf.reduce_sum(tf.abs(outputs_original - outputs_loaded))
        # print(f"Sum of output differences: {diff_sum}")
        self.assertLess(diff_sum, 1e-5, "Output difference is too large")
        
        # Check if optimizer configuration was loaded correctly
        optimizer_config = loaded_model_a.optimizer.get_config()
        # print("Loaded model optimizer config:", optimizer_config)
        self.assertEqual(optimizer_config['name'], 'adam', "Incorrect optimizer type")
    
    def test_model_structure(self):
        """Test if model structure and nested relationships are correctly saved and loaded"""
        # Create original model
        original_model = A2()

        original_model.compile(optimizer='adam', loss='mse')
        original_model.fit(self.x_train, self.y_train, epochs=3, verbose=0)

        # Save the model
        original_model.save(self.model_path)

        from tensorflow.keras.utils import get_custom_objects
        cur_dict = {
            'A2': A2,
            'B2': B2,
            'C2': C2,  
            'nn_Linear': nn_Linear,
            'nn_ReLU': nn_ReLU,
        }
        get_custom_objects().update(cur_dict)
        loaded_model = tf.keras.models.load_model(self.model_path ,  custom_objects=get_custom_objects())
        

        # Verify model hierarchy
        # Model A should have dense_a, relu, and submodel B
        self.assertTrue(hasattr(loaded_model, 'dense_a'), "Loaded model missing dense_a layer")
        self.assertTrue(hasattr(loaded_model, 'relu'), "Loaded model missing relu layer")
        self.assertTrue(hasattr(loaded_model, 'b'), "Loaded model missing submodel B")
        
        # Submodel B should have dense_b, relu, and submodel C
        self.assertTrue(hasattr(loaded_model.b, 'dense_b'), "Submodel B missing dense_b layer")
        self.assertTrue(hasattr(loaded_model.b, 'relu'), "Submodel B missing relu layer")
        self.assertTrue(hasattr(loaded_model.b, 'c'), "Submodel B missing submodel C")
        
        # Submodel C should have dense_c and relu
        self.assertTrue(hasattr(loaded_model.b.c, 'dense_c'), "Submodel C missing dense_c layer")
        self.assertTrue(hasattr(loaded_model.b.c, 'relu'), "Submodel C missing relu layer")
        
        # Verify configuration parameters
        self.assertEqual(loaded_model.units, 16, "Incorrect units parameter in model A")
        self.assertEqual(loaded_model.b.units, 8, "Incorrect units parameter in submodel B")
        self.assertEqual(loaded_model.b.c.units, 4, "Incorrect units parameter in submodel C")


if __name__ == '__main__':
    unittest.main()
