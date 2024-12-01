"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main

"""


import tensorflow as tf
import numpy as np

# class SpatialEmb(nn.Module):
class SpatialEmb(tf.keras.layers.Layer):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):

        print("modules.py: SpatialEmb.__init__()")

        super().__init__()

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        #输入是proj_in_dim维度的
        # Input projection layers
        self.input_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(proj_dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.ReLU()
        ])

        # Learnable weights
        self.weight = self.add_weight(
            shape=(1, num_proj, proj_dim),
            initializer="random_normal",
            trainable=True,
            name="weight"
        )
        self.dropout = tf.keras.layers.Dropout(dropout)


    # def extra_repr(self) -> str:

    #     print("modules.py: SpatialEmb.extra_repr()")

    #     return f"weight: nn.Parameter ({self.weight.size()})"
    def __repr__(self):
        weight_shape = self.weight.shape
        return f"SpatialEmb(weight: tf.Variable {weight_shape})"


    def call(self, feat, prop, training=False):

        print("modules.py: SpatialEmb.call()")

        # Transpose dimensions for TensorFlow
        feat = tf.transpose(feat, perm=[0, 2, 1])

        if self.prop_dim > 0 and prop is not None:
            repeated_prop = tf.expand_dims(prop, axis=1)
            repeated_prop = tf.tile(repeated_prop, [1, tf.shape(feat)[1], 1])
            feat = tf.concat([feat, repeated_prop], axis=-1)

        y = self.input_proj(feat, training=training)
        z = tf.reduce_sum(self.weight * y, axis=1)
        z = self.dropout(z, training=training)
        return z



def replicate_pad(x, pad):
    # Extract dimensions
    batch, height, width, channels = x.shape

    # Pad height (top and bottom)
    top = tf.repeat(x[:, :1, :, :], repeats=pad, axis=1)  # Replicate the first row `pad` times
    bottom = tf.repeat(x[:, -1:, :, :], repeats=pad, axis=1)  # Replicate the last row `pad` times
    x = tf.concat([top, x, bottom], axis=1)

    # Pad width (left and right)
    left = tf.repeat(x[:, :, :1, :], repeats=pad, axis=2)  # Replicate the first column `pad` times
    right = tf.repeat(x[:, :, -1:, :], repeats=pad, axis=2)  # Replicate the last column `pad` times
    x = tf.concat([left, x, right], axis=2)

    return x


class RandomShiftsAug:
    def __init__(self, pad):
        
        print("modules.py: RandomShiftsAug.__init__()")

        self.pad = pad


    def __call__(self, x):

        print("modules.py: RandomShiftsAug.__call__()")

        n, c, h, w = x.size()
        assert h == w

        # # Add padding with replication
        x = replicate_pad(x, self.pad)

        # Create a random shift grid
        eps = 1.0 / (h + 2 * self.pad)
        arange = tf.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad)[:h]

        arange = tf.reshape(arange, (1, h, 1))
        arange = tf.tile(arange, [h, 1, 1])


        base_grid = tf.concat([arange, tf.transpose(arange, perm=[1, 0, 2])], axis=-1)
        base_grid = tf.expand_dims(base_grid, axis=0)
        base_grid = tf.tile(base_grid, [n, 1, 1, 1])

        shift = tf.random.uniform(
            shape=(n, 1, 1, 2), minval=0, maxval=2 * self.pad + 1, dtype=tf.float32
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        
        return tf.nn.grid_sample(x, grid, padding_mode="zeros", align_corners=False)




# test random shift
if __name__ == "__main__":

    print("modules.py: main()")

    from PIL import Image
    import requests
    import numpy as np

    image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((96, 96))


    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.transpose(image, perm=[2, 0, 1, 3])  # Convert to NHWC format

    aug = RandomShiftsAug(pad=4)
    image_aug = aug(image)
    image_aug = tf.squeeze(image_aug)

    image = tf.transpose(image, perm=[1, 2, 0, 3])
    image_aug = image_aug.numpy()
    image_aug = Image.fromarray(image_aug.astype(np.uint8))

    image_aug.show()
























