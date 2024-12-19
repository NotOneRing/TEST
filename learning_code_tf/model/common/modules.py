"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main

"""


import tensorflow as tf
import numpy as np

from util.torch_to_tf import nn_Sequential, nn_Linear,\
nn_LayerNorm, nn_ReLU, nn_Parameter, torch_zeros, nn_Dropout,\
torch_tensor_transpose, torch_cat, torch_tensor_repeat, torch_unsqueeze,\
torch_sum, torch_tensor

# class SpatialEmb(nn.Module):
class SpatialEmb(tf.keras.layers.Layer):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):

        print("modules.py: SpatialEmb.__init__()")

        super().__init__()

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        # #输入是proj_in_dim维度的
        # # Input projection layers
        # self.input_proj = tf.keras.Sequential([
        #     tf.keras.layers.Dense(proj_dim),
        #     tf.keras.layers.LayerNormalization(),
        #     tf.keras.layers.ReLU()
        # ])

        #输入是proj_in_dim维度的
        # Input projection layers
        self.input_proj = nn_Sequential([
            nn_Linear(proj_in_dim, proj_dim),
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

        self.weight = nn_Parameter(torch_zeros(1, num_proj, proj_dim))
        self.dropout = nn_Dropout(dropout)


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



from util.torch_to_tf import nn_functional_pad_replicate, torch_linspace,\
torch_randint

class RandomShiftsAug:
    def __init__(self, pad):
        
        print("modules.py: RandomShiftsAug.__init__()")

        self.pad = pad


    def __call__(self, x):

        print("modules.py: RandomShiftsAug.__call__()")

        # n, c, h, w = x.size()
        n, c, h, w = x.shape

        assert h == w

        # # Add padding with replication
        x = nn_functional_pad_replicate(x, self.pad)

        # Create a random shift grid
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch_linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad)[:h]

        # arange = tf.reshape(arange, (1, h, 1))
        # arange = tf.tile(arange, [h, 1, 1])
        arange = torch_unsqueeze( torch_tensor_repeat( torch_unsqueeze(arange, 0), h, 1), 2)


        # base_grid = tf.concat([arange, tf.transpose(arange, perm=[1, 0, 2])], axis=-1)
        # base_grid = tf.expand_dims(base_grid, axis=0)
        # base_grid = tf.tile(base_grid, [n, 1, 1, 1])


        base_grid = torch_cat([arange, torch_tensor_transpose(arange, 1, 0)], dim=2)

        base_grid = torch_tensor_repeat( torch_unsqueeze(base_grid, 0), n, 1, 1, 1)

        # shift = tf.random.uniform(
        #     shape=(n, 1, 1, 2), minval=0, maxval=2 * self.pad + 1, dtype=tf.float32
        # )
        # shift *= 2.0 / (h + 2 * self.pad)
        shift = torch_randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        
        return tf.nn.grid_sample(x, grid, padding_mode="zeros", align_corners=False)




# # test random shift
# if __name__ == "__main__":

#     print("modules.py: main()")

#     from PIL import Image
#     import requests
#     import numpy as np

#     image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
#     image = Image.open(requests.get(image_url, stream=True).raw)
#     image = image.resize((96, 96))


#     image = tf.convert_to_tensor(image, dtype=tf.float32)
#     image = tf.expand_dims(image, axis=0)  # Add batch dimension
#     image = tf.transpose(image, perm=[2, 0, 1, 3])  # Convert to NHWC format

#     aug = RandomShiftsAug(pad=4)
#     image_aug = aug(image)
#     image_aug = tf.squeeze(image_aug)

#     image = tf.transpose(image, perm=[1, 2, 0, 3])
#     image_aug = image_aug.numpy()
#     image_aug = Image.fromarray(image_aug.astype(np.uint8))

#     image_aug.show()




# test random shift
if __name__ == "__main__":

    print("modules.py: main()", flush = True)

    from PIL import Image
    import requests
    import numpy as np

    image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((96, 96))

    image = torch_tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
    
    aug = RandomShiftsAug(pad=4)
    image_aug = aug(image)

    image_aug = image_aug.squeeze().permute(1, 2, 0).numpy()
    image_aug = Image.fromarray(image_aug.astype(np.uint8))
    image_aug.show()































