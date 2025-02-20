"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main

"""


import tensorflow as tf
import numpy as np

from util.torch_to_tf import nn_Sequential, nn_Linear,\
nn_LayerNorm, nn_ReLU, nn_Parameter, torch_zeros, nn_Dropout,\
torch_tensor_transpose, torch_cat, torch_tensor_repeat, torch_unsqueeze,\
torch_sum, torch_tensor, save_tf_Variable, load_tf_Variable

from util.config import DATASET_NAME


# class SpatialEmb(nn.Module):
class SpatialEmb(tf.keras.layers.Layer):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout, serialized_input_proj = None, serialized_dropout = None, weight = None, **kwargs):

        print("modules.py: SpatialEmb.__init__()")

        super().__init__()

        self.num_patch = num_patch
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim
        self.proj_dim = proj_dim

        self.initial_dropout = dropout

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim


        # #输入是proj_in_dim维度的
        # # Input projection layers
        # self.input_proj = tf.keras.Sequential([
        #     tf.keras.layers.Dense(proj_dim),
        #     tf.keras.layers.LayerNormalization(),
        #     tf.keras.layers.ReLU()
        # ])

        #输入是proj_in_dim维度的
        # Input projection layers
        if serialized_input_proj:
            self.input_proj = serialized_input_proj
        else:
            self.input_proj = nn_Sequential([
                nn_Linear(proj_in_dim, proj_dim, name_Dense="SpatialEmb_input_proj_1"),
                nn_LayerNorm(proj_dim),
                nn_ReLU(inplace=True)
            ])
        # # Learnable weights
        # self.weight = self.add_weight(
        #     shape=(1, num_proj, proj_dim),
        #     initializer="random_normal",
        #     trainable=True,
        #     name="weight"
        # )

        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn_Parameter(torch_zeros(1, num_proj, proj_dim))


        if serialized_dropout:
            self.dropout = serialized_dropout
        else:
            self.dropout = nn_Dropout(dropout)
        
        
        
        
    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(SpatialEmb, self).get_config()


        print(f"num_patch: {self.num_patch}, type: {type(self.num_patch)}")
        print(f"patch_dim: {self.patch_dim}, type: {type(self.patch_dim)}")
        print(f"prop_dim: {self.prop_dim}, type: {type(self.prop_dim)}")
        print(f"proj_dim: {self.proj_dim}, type: {type(self.proj_dim)}")
        print(f"dropout: {self.dropout}, type: {type(self.dropout)}")

        # if hasattr(self, 'serialized_input_proj'):
        #     print(f"serialized_input_proj: {self.serialized_input_proj}, type: {type(self.serialized_input_proj)}")

        config.update({
        "serialized_input_proj": tf.keras.layers.serialize(self.input_proj),
        })
        # else:
        #     config.update({
        #     "serialized_input_proj": None,
        #     })
            

        # if hasattr(self, 'serialized_dropout'):
        #     print(f"serialized_dropout: {self.serialized_dropout}, type: {type(self.serialized_dropout)}")

        config.update({
        "serialized_dropout": tf.keras.layers.serialize(self.dropout)
        })
        # else:
        #     config.update({
        #     "serialized_dropout": None,
        #     })

        save_tf_Variable(self.weight, "SpatialEmb_weight" + DATASET_NAME)

        config.update({
                        "num_patch": self.num_patch,
                        "patch_dim": self.patch_dim,
                        "prop_dim": self.prop_dim,
                        "proj_dim": self.proj_dim,
                        "dropout": self.initial_dropout,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""

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
            'MLP': MLP,                            # 自定义的 MLP 层
            'ResidualMLP': ResidualMLP,            # 自定义的 ResidualMLP 层
            'nn_Sequential': nn_Sequential,        # 自定义的 Sequential 类
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

        config_serialized_input_proj = config.pop("serialized_input_proj")
        if config_serialized_input_proj:
            serialized_input_proj = tf.keras.layers.deserialize( config_serialized_input_proj,  custom_objects=get_custom_objects() )
        else:
            serialized_input_proj = None
        config_serialized_dropout = config.pop("serialized_dropout")
        if config_serialized_dropout:
            serialized_dropout = tf.keras.layers.deserialize( config_serialized_dropout,  custom_objects=get_custom_objects() )
        else:
            serialized_dropout = None

        weight = load_tf_Variable("SpatialEmb_weight" + DATASET_NAME)

        return cls(serialized_input_proj=serialized_input_proj, serialized_dropout=serialized_dropout, weight=weight, **config)


    # def extra_repr(self) -> str:

    #     print("modules.py: SpatialEmb.extra_repr()")

    #     return f"weight: nn.Parameter ({self.weight.size()})"
    def __repr__(self):
        weight_shape = self.weight.shape
        return f"SpatialEmb(weight: tf.Variable {weight_shape})"


    def call(self, feat, prop, training=False):

        print("modules.py: SpatialEmb.call()")

        # Transpose dimensions for TensorFlow
        # feat = tf.transpose(feat, perm=[0, 2, 1])
        feat = torch_tensor_transpose(feat, 1, 2)

        if self.prop_dim > 0 and prop is not None:
            # repeated_prop = tf.expand_dims(prop, axis=1)
            # repeated_prop = tf.tile(repeated_prop, [1, tf.shape(feat)[1], 1])

            repeated_prop = torch_tensor_repeat( torch_unsqueeze(prop, 1), 1, feat.shape[1], 1)

            feat = torch_cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat, training=training)
        z = torch_sum(self.weight * y, dim=1)
        z = self.dropout(z, training=training)
        return z



from util.torch_to_tf import nn_functional_pad, torch_linspace,\
torch_randint, torch_nn_functional_grid_sample

class RandomShiftsAug(tf.keras.layers.Layer):
    def __init__(self, pad, **kwargs):
        
        print("modules.py: RandomShiftsAug.__init__()")
        super().__init__(**kwargs)  # Ensure parent class is initialized

        self.pad = pad


    def get_config(self):
        """Returns the config of the layer for serialization."""
        config = super(RandomShiftsAug, self).get_config()
        config.update({
                        "pad": self.pad,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates the layer from its config."""
        return cls(**config)
    

    def call(self, x):

        print("modules.py: RandomShiftsAug.__call__()")

        # n, c, h, w = x.size()
        n, c, h, w = x.shape

        assert h == w
        padding = tuple([self.pad] * 4)

        print("modules.py: RandomShiftsAug.__call__(): 1")

        # # Add padding with replication
        x = nn_functional_pad(x, padding, "replicate")

        print("modules.py: RandomShiftsAug.__call__(): 2")

        # Create a random shift grid
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch_linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad)[:h]

        print("modules.py: RandomShiftsAug.__call__(): 3")

        arange = torch_unsqueeze( torch_tensor_repeat( torch_unsqueeze(arange, 0), h, 1), 2)

        print("modules.py: RandomShiftsAug.__call__(): 4")



        base_grid = torch_cat([arange, torch_tensor_transpose(arange, 1, 0)], dim=2)

        print("modules.py: RandomShiftsAug.__call__(): 5")

        base_grid = torch_tensor_repeat( torch_unsqueeze(base_grid, 0), n, 1, 1, 1)

        print("modules.py: RandomShiftsAug.__call__(): 6")


        shift = torch_randint(
            low = 0, high = 2 * self.pad + 1, size=(n, 1, 1, 2), dtype=x.dtype
        )

        print("modules.py: RandomShiftsAug.__call__(): 7")

        shift *= 2.0 / (h + 2 * self.pad)

        print("modules.py: RandomShiftsAug.__call__(): 8")

        grid = base_grid + shift
        
        print("modules.py: RandomShiftsAug.__call__(): 9")
        
        result = torch_nn_functional_grid_sample(x, grid, padding_mode="zeros", align_corners=False)

        print("modules.py: RandomShiftsAug.__call__(): 9")

        return result




from util.torch_to_tf import torch_tensor_permute, torch_unsqueeze, torch_tensor_float, torch_squeeze

# test random shift
if __name__ == "__main__":

    print("modules.py: main()", flush = True)

    from PIL import Image
    import requests
    import numpy as np

    image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((96, 96))

    # image = torch_tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
    image = torch_tensor_float( torch_unsqueeze(torch_tensor_permute( torch_tensor(np.array(image)), 2, 0, 1), 0) )

    
    aug = RandomShiftsAug(pad=4)
    image_aug = aug(image)

    image_matrix = np.array(image_aug)
    print("Shape of the image matrix:", image_matrix.shape)



    image_aug = torch_tensor_permute( torch_squeeze( image_aug ), 1, 2, 0).numpy()
    image_aug = Image.fromarray(image_aug.astype(np.uint8))
    image_aug.show()

    image_aug.save("augmented_image.jpg", format="JPEG")

    # # Convert to NumPy array (matrix)
    # image_matrix = np.array(image_aug)

    # # Print the shape and matrix
    # print("Shape of the image matrix:", image_matrix.shape)

    # # Loop over pixels and print values
    # for i in range(image_matrix.shape[0]):  # Loop over height
    #     for j in range(image_matrix.shape[1]):  # Loop over width
    #         pixel = image_matrix[i, j]
    #         print(f"Pixel at ({i}, {j}): {pixel}")































