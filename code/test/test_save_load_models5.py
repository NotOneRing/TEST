import tensorflow as tf
import numpy as np
from tensorflow.keras.saving import register_keras_serializable


from model.diffusion.mlp_diffusion import DiffusionMLP
from model.diffusion.diffusion import DiffusionModel
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug
from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish

from tensorflow.keras.utils import get_custom_objects

cur_dict = {
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
# Register your custom class with Keras
get_custom_objects().update(cur_dict)

# print('get_custom_objects() = ', get_custom_objects())

print("Custom objects:", get_custom_objects())
assert 'SinusoidalPosEmb' in get_custom_objects()



serialized =  {'name': 'SinusoidalPosEmb', 'trainable': True, 'dtype': {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}, 'registered_name': None}, 'dim': 32}


sinu = SinusoidalPosEmb.from_config(serialized)

print("sinu = ", sinu)

result = tf.keras.layers.deserialize( serialized ,  custom_objects=get_custom_objects() )

print("result = ", result)


