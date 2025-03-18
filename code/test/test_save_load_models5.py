import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import get_custom_objects

from model.diffusion.mlp_diffusion import DiffusionMLP
from model.diffusion.diffusion import DiffusionModel
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug
from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish


class TestSaveLoadModels(unittest.TestCase):
    """Test class for verifying serialization and deserialization of custom Keras objects."""
    
    def setUp(self):
        """Set up the test environment by registering custom objects with Keras."""
        self.custom_objects = {
            'DiffusionModel': DiffusionModel,  # Register the custom DiffusionModel class
            'DiffusionMLP': DiffusionMLP,
            # 'VPGDiffusion': VPGDiffusion,
            'SinusoidalPosEmb': SinusoidalPosEmb,   
            'MLP': MLP,                            # Custom MLP layer
            'ResidualMLP': ResidualMLP,            # Custom ResidualMLP layer
            'nn_Sequential': nn_Sequential,        # Custom Sequential class
            'nn_Linear': nn_Linear,
            'nn_LayerNorm': nn_LayerNorm,
            'nn_Dropout': nn_Dropout,
            'nn_ReLU': nn_ReLU,
            'nn_Mish': nn_Mish,
            'SpatialEmb': SpatialEmb,
            'RandomShiftsAug': RandomShiftsAug,
        }
        # Register custom objects with Keras
        get_custom_objects().update(self.custom_objects)
    
    def test_custom_objects_registration(self):
        """Test that custom objects are properly registered with Keras."""
        # Verify custom objects are registered
        self.assertIn('SinusoidalPosEmb', get_custom_objects())
        # print("Custom objects:", get_custom_objects())
    
    def test_sinusoidal_pos_emb_serialization(self):
        """Test serialization and deserialization of SinusoidalPosEmb class."""
        # Define serialized configuration for SinusoidalPosEmb
        serialized = {
            'name': 'SinusoidalPosEmb', 
            'trainable': True, 
            'dtype': {
                'module': 'keras', 
                'class_name': 'DTypePolicy', 
                'config': {'name': 'float32'}, 
                'registered_name': None
            }, 
            'dim': 32
        }
        
        # Test from_config method
        sinu = SinusoidalPosEmb.from_config(serialized)
        # print("sinu =", sinu)
        self.assertIsInstance(sinu, SinusoidalPosEmb)
        self.assertEqual(sinu.dim, 32)

        serialized = tf.keras.layers.serialize(sinu)
        
        # Test deserialize method
        result = tf.keras.layers.deserialize(serialized, custom_objects=get_custom_objects())
        # print("result =", result)
        self.assertIsInstance(result, SinusoidalPosEmb)
        self.assertEqual(result.dim, 32)


if __name__ == '__main__':
    unittest.main()
