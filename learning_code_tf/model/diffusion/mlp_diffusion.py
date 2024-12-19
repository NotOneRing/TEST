import tensorflow as tf
import numpy as np
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug

# from tensorflow.keras.layers import Layers

log = tf.get_logger()

from util.torch_to_tf import nn_Sequential, nn_Linear, nn_LayerNorm, nn_Dropout, nn_ReLU, nn_Mish

from util.torch_to_tf import torch_tensor_float, torch_cat, torch_flatten, torch_view, torch_reshape

# class VisionDiffusionMLP(tf.keras.layers.Layer):
class VisionDiffusionMLP(tf.keras.Model):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        print("mlp_diffusion.py: VisionDiffusionMLP.__init__()")

        super(VisionDiffusionMLP, self).__init__()

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn_Sequential([
                nn_Linear(self.backbone.repr_dim, visual_feature_dim),
                nn_LayerNorm(),
                nn_Dropout(dropout),
                nn_ReLU(),
            ])

        # diffusion
        input_dim = time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn_Sequential([
            SinusoidalPosEmb(time_dim),
            nn_Linear(time_dim, time_dim * 2),
            nn_Mish(),
            nn_Linear(time_dim * 2, time_dim),
        ])

        if residual_style:
            model = ResidualMLP
        else:
            model = MLP

        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def call(self, x, time, cond, **kwargs):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        """

        print("mlp_diffusion.py: VisionDiffusionMLP.call()")

        B, Ta, Da = x.shape
        _, T_rgb, C, H, W = cond["rgb"].shape

        # flatten chunk
        x = tf.reshape(x, [B, -1])

        # flatten history
        state = tf.reshape(cond["state"], [B, -1])

        # Take recent images
        rgb = cond["rgb"][:, -self.img_cond_steps:]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = torch_reshape(rgb, [B, T_rgb, self.num_img, 3, H, W])
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = torch_tensor_float(rgb)

        # get vit output - pass in two images separately
        if self.num_img > 1:
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1(feat1, state)
            feat2 = self.compress2(feat2, state)
            feat = torch_cat([feat1, feat2], axis=-1)
        else:
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.call(feat, state)
            else:
                # feat = torch_reshape(feat, [B, -1])
                feat = torch_flatten(feat, 1, -1)
                feat = self.compress(feat)

        cond_encoded = tf.concat([feat, state], dim=-1)

        # append time and cond
        time = torch_view(time, [B, 1])
        time_emb = torch_view( self.time_embedding(time), B, self.time_dim )
        x = torch_cat([x, time_emb, cond_encoded], dim=-1)

        # mlp
        out = self.mlp_mean(x)
        return torch_view(out, [B, Ta, Da])


import math


class KaimingUniformInitializer(tf.keras.initializers.Initializer):
    def __init__(self, fan_in):
        self.fan_in = fan_in

    def __call__(self, shape, dtype=None):
        limit = math.sqrt(3.0 / self.fan_in)  # PyTorch 的 Kaiming Uniform 范围
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

    def get_config(self):  # 必须实现以支持序列化
        return {'fan_in': self.fan_in}


class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.kernel_initializer = KaimingUniformInitializer(fan_in=input_dim)
        self.bias_initializer = tf.keras.initializers.Zeros()

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=self.kernel_initializer,
            name='kernel',
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            name='bias',
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            "units": self.units,
            "input_dim": self.input_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

































# class DiffusionMLP(tf.keras.layers.Layer):
class DiffusionMLP(tf.keras.Model):

    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        print("mlp_diffusion.py: DiffusionMLP.__init__()")

        super(DiffusionMLP, self).__init__()

        print("before sinusiodalPosEmb()")

        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.cond_dim = cond_dim

        print("self.cond_dim = ", self.cond_dim)

        # print("self.time_dim = ", time_dim)

        self.time_dim = time_dim
        self.mlp_dims = mlp_dims
        self.cond_mlp_dims = cond_mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style


        self.output_dim = self.action_dim * self.horizon_steps

        print("self.time_dim = ", self.time_dim)

        self.time_embedding = nn_Sequential([
            SinusoidalPosEmb(time_dim),
            nn_Linear(time_dim, time_dim * 2),
            nn_Mish(),
            nn_Linear(time_dim * 2, time_dim),        
        ])


        # self.time_embedding = tf.keras.Sequential([
        #     SinusoidalPosEmb(time_dim),
        #     CustomDense(units=time_dim * 2, input_dim=time_dim),  # 自定义初始化的 Dense 层
        #     tf.keras.layers.Activation('mish'),
        #     CustomDense(units=time_dim, input_dim=time_dim * 2),  # 自定义初始化的 Dense 层
        # ])




        # print("time_dim = ", self.time_dim)

        



        # print("after sinusiodalPosEmb()")
        
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP


        # print("after ResidualMLP and MLP")
        

        if self.cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [self.cond_dim] + self.cond_mlp_dims,
                activation_type=self.activation_type,
                out_activation_type="Identity",
            )
            self.input_dim = self.time_dim + self.action_dim * self.horizon_steps + self.cond_mlp_dims[-1]
        else:
            self.input_dim = self.time_dim + self.action_dim * self.horizon_steps + self.cond_dim


        # print("after cond_mlp and input_dim")


        # print("[self.input_dim] + self.mlp_dims + [self.output_dim] = ", [self.input_dim] + self.mlp_dims + [self.output_dim])
        

        self.mlp_mean = model(
            [self.input_dim] + self.mlp_dims + [self.output_dim],
            activation_type=self.activation_type,
            out_activation_type=self.out_activation_type,
            use_layernorm=self.use_layernorm,
        )


        print("after mlp_mean")
        

        # self.time_dim = time_dim




    # def call(self, x, time, cond, **kwargs):
    def call(self, x, time, state, **kwargs):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """

        # print("mlp_diffusion.py: DiffusionMLP.call()")
        # print("x.shape = ", x.shape)
        # print("time.shape = ", time.shape)
        # • cond={'state': 'tf.Tensor(shape=(128, 1, 11), dtype=float32)'}
        # • kwargs={'training': 'True'}
        # print("cond.shape = ", cond.shape)

        B, Ta, Da = x.shape
        # B, Ta, Da = x_shape

        assert B == x.shape[0]
        assert Ta == self.horizon_steps
        assert Da == self.action_dim

        x = torch_view(x, B, -1)

        # state = torch_view(cond["state"], B, -1)
        # # flatten chunk
        # x = tf.reshape(x, [B, -1])

        # # flatten history
        # state = tf.reshape(cond["state"], [B, -1])

        # # append time and cond
        # time = tf.reshape(time, [B, 1])

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)


        # print("B = ", B)

        # print("time = ", time)

        # print("state = ", state)
        
        # print("1time.shape = ", time.shape)

        # time = tf.squeeze(time, axis=1)
        # time = tf.squeeze(time, axis=-1)
        # time = tf.reshape(time, [B])

        # print("2time.shape = ", time.shape)


        time_emb = self.time_embedding(time)

        # print("time_emb = ", time_emb)

        # print("time_emb.shape = ", time_emb.shape)

        time_emb = tf.squeeze(time_emb, axis=1)

        # print("after tf.squeeze")

        # for layer in self.time_embedding.layers:
        #     if isinstance(layer, CustomDense):
        #         print("TensorFlow Dense weights:", layer.kernel.numpy())
        #         print("TensorFlow Dense bias:", layer.bias.numpy())


        # print("x = ", x)

        # print("time_emb = ", time_emb)

        # print("state = ", state)
        

        # print("x.shape = ", x.shape)

        # print("time_emb.shape = ", time_emb.shape)

        # print("state.shape = ", state.shape)


        x = tf.concat([x, time_emb, state], axis=-1)

        # mlp head
        out = self.mlp_mean(x)


        # print("out.shape[0] = ", out.shape[0])
        # print("out.shape = ", out.shape)

        if out.shape[0]:
            # print("branch1")
            # print("out.shape[0] = ", out.shape[0])
            final_out = tf.reshape(out, [-1, Ta, Da])
        else:
            # print("branch2")
            # print("out.shape[0] = ", out.shape[0])
            # final_out = tf.reshape(out, [None, Ta, Da])
            final_out = tf.reshape(out, [-1, Ta, Da])

        # print("final_out.shape = ", final_out.shape)

        # return tf.reshape(out, [B, Ta, Da])
        return final_out


    
    # def summary(self):
    #     from tensorflow.keras import Input, Model

    #     if self.cond_mlp_dims is not None:
    #         input_dim = self.time_dim + self.action_dim * self.horizon_steps + self.cond_mlp_dims[-1]
    #     else:
    #         input_dim = self.time_dim + self.action_dim * self.horizon_steps + self.cond_dim

    #     input_shape = (input_dim,)  

    #     x = Input(shape=input_shape)

    #     output = self.call(x)

    #     model = Model(inputs=x, outputs=output)

    #     # 返回模型的 summary
    #     return model.summary()


    def summary(self):
        from tensorflow.keras import Input, Model

        # 假设x的形状为 (self.action_dim * self.horizon_steps,)
        x_input = Input(shape=(self.action_dim * self.horizon_steps,), name='x_input')

        # print("self.action_dim * self.horizon_steps = ", self.action_dim * self.horizon_steps)

        # 假设time的形状为 (time_dim,) 
        time_input = Input(shape=(1,), name='time_input')


        if self.cond_mlp_dims is not None:
            # 假设cond的形状为 (cond_dim,)
            cond_input = Input(shape=(self.cond_mlp_dims[-1],), name='cond_input')
            # print("self.cond_mlp_dims[-1] = ", self.cond_mlp_dims[-1])
        else:
            cond_input = Input(shape=(self.cond_dim,), name='cond_input')
            # print("self.cond_dim = ", self.cond_dim)

        # print("[x_input, time_input, cond_input] = ", [x_input, time_input, cond_input])

        # 调用模型的 call 方法获取输出
        output = self.call(x_input, time_input, cond_input)


        # 创建模型
        model = Model(inputs=[x_input, time_input, cond_input], outputs=output)

        # 返回模型的 summary
        return model.summary()



    def get_config(self):

        print("DiffusionMLP: get_config()")

        config = super(DiffusionMLP, self).get_config()

        # 打印每个属性及其类型和值
        print("Checking DiffusionMLP Config elements:")
        print(f"action_dim: {self.action_dim}, type: {type(self.action_dim)}")
        print(f"horizon_steps: {self.horizon_steps}, type: {type(self.horizon_steps)}")
        print(f"cond_dim: {self.cond_dim}, type: {type(self.cond_dim)}")
        print(f"time_dim: {self.time_dim}, type: {type(self.time_dim)}")
        print(f"mlp_dims: {self.mlp_dims}, type: {type(self.mlp_dims)}")
        print(f"cond_mlp_dims: {self.cond_mlp_dims}, type: {type(self.cond_mlp_dims)}")
        print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
        print(f"out_activation_type: {self.out_activation_type}, type: {type(self.out_activation_type)}")
        print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
        print(f"residual_style: {self.residual_style}, type: {type(self.residual_style)}")
        print(f"output_dim: {self.output_dim}, type: {type(self.output_dim)}")
        print(f"input_dim: {self.input_dim}, type: {type(self.input_dim)}")
        
        config.update({
            "action_dim": self.action_dim,
            "horizon_steps": self.horizon_steps,
            "cond_dim": self.cond_dim,
            "time_dim": self.time_dim,
            "mlp_dims": self.mlp_dims,
            "cond_mlp_dims": self.cond_mlp_dims,
            "activation_type": self.activation_type,
            "out_activation_type": self.out_activation_type,
            "use_layernorm": self.use_layernorm,
            "residual_style": self.residual_style,
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "time_dim": self.time_dim,
        })
        return config



    @classmethod
    def from_config(cls, config):
        return cls(**config)






















