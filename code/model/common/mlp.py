

import tensorflow as tf
from tensorflow.keras import models

from collections import OrderedDict
import logging

from util.torch_to_tf import nn_ReLU, nn_GELU, nn_Tanh, nn_ELU, nn_Mish, nn_Identity, nn_Softplus


from util.config import DEBUG, TEST_LOAD_PRETRAIN, OUTPUT_VARIABLES, OUTPUT_POSITIONS, OUTPUT_FUNCTION_HEADER


activation_dict = {
    "ReLU": nn_ReLU(),
    "ELU": nn_ELU(),
    "GELU": nn_GELU(),
    "Tanh": nn_Tanh(),
    "Mish": nn_Mish(),
    "Identity": nn_Identity(),
    "Softplus": nn_Softplus(),
}


from util.torch_to_tf import nn_ModuleList, nn_Linear, nn_Sequential, nn_LayerNorm, nn_Dropout, torch_cat

from tensorflow.keras.saving import register_keras_serializable
@register_keras_serializable(package="Custom")
class MLP(
    tf.keras.layers.Layer
    ):


    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        verbose=False,
        moduleList = None,
        name = "MLP",
        **kwargs
    ):
        if OUTPUT_FUNCTION_HEADER:
            print("mlp.py: MLP.__init__()", flush = True)

        super(MLP, self).__init__(name=name, **kwargs)


        self.dim_list = list(dim_list)

        self.append_dim=append_dim
        self.append_layers=append_layers
        self.activation_type=activation_type
        self.out_activation_type=out_activation_type
        self.use_layernorm=use_layernorm
        self.use_layernorm_final=use_layernorm_final
        self.dropout = dropout
        self.use_drop_final=use_drop_final
        self.verbose=verbose

        # Construct module list: if use `Python List`, the modules are not
        # added to computation graph. Instead, we should use `nn.ModuleList()`.
        if moduleList == None:
            self.moduleList = []

            num_layer = len(dim_list) - 1

            if OUTPUT_POSITIONS:
                print("MLP __init__(): moduleList == None 1")
            if OUTPUT_VARIABLES:
                print("num_layer = ", num_layer)
            for idx in range(num_layer):
                i_dim = dim_list[idx]
                o_dim = dim_list[idx + 1]
                if append_dim > 0 and idx in append_layers:
                    i_dim += append_dim
                linear_layer = nn_Linear(i_dim, o_dim, name_Dense="MLP_linear" + str(idx) + "-1")

                # Add module components
                layers = [("linear_1", linear_layer)]
                if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                    layers.append(("norm_1", nn_LayerNorm(o_dim)))
                if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                    layers.append(("dropout_1", nn_Dropout(dropout)))

                # add activation function
                act = (
                    activation_dict[activation_type]
                    if idx != num_layer - 1
                    else activation_dict[out_activation_type]
                )
                layers.append(("act_1", act))

                # re-construct module
                
                module = nn_Sequential(OrderedDict(layers), name = "MLP_module" + str(idx) + "-1")

                current_config = module.get_config()

                if OUTPUT_VARIABLES:
                    print("current_config = ", current_config)

                    for i, layer in enumerate(layers):
                        print("layer ", i)
                        print("layer[1] = ", layer[1])
                        print("layer[1].get_config() = ", layer[1].get_config())


                self.moduleList.append(module)

            if OUTPUT_POSITIONS:
                print("MLP __init__(): moduleList == None 1")
            
            self.moduleList = nn_Sequential(self.moduleList, name="MLP_moduleList")

        else:
            if OUTPUT_POSITIONS:
                print("MLP __init__(): moduleList != None 1")
            self.moduleList = moduleList
            if OUTPUT_POSITIONS:
                print("MLP __init__(): moduleList != None 1")
        if verbose:
            logging.info(self.moduleList)


    def get_config(self):
        config = super(MLP, self).get_config()  # Call the base class's get_config


        print("Checking MLP Config elements:")
        print(f"dim_list: {self.dim_list}, type: {type(self.dim_list)}")
        print(f"append_dim: {self.append_dim}, type: {type(self.append_dim)}")
        print(f"append_layers: {self.append_layers}, type: {type(self.append_layers)}")
        print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
        
        print(f"out_activation_type: {self.out_activation_type}, type: {type(self.out_activation_type)}")
        print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
        print(f"use_layernorm_final: {self.use_layernorm_final}, type: {type(self.use_layernorm_final)}")


        print(f"dropout: {self.dropout}, type: {type(self.dropout)}")

        print(f"use_drop_final: {self.use_drop_final}, type: {type(self.use_drop_final)}")

        print(f"verbose: {self.verbose}, type: {type(self.verbose)}")

        print(f"moduleList: {self.moduleList}, type: {type(self.moduleList)}")

        config.update({
            "dim_list": self.dim_list,
            "append_dim": self.append_dim,
            "append_layers": self.append_layers,
            "activation_type": self.activation_type,
            "out_activation_type": self.out_activation_type,
            "use_layernorm": self.use_layernorm,
            "use_layernorm_final": self.use_layernorm_final,
            "dropout": self.dropout,
            "use_drop_final": self.use_drop_final,
            "verbose": self.verbose,
            "moduleList": self.moduleList.get_config()
        })

        if OUTPUT_VARIABLES:
            print("MLP: config = ", config)

        return config
    

    @classmethod
    def from_config(cls, config):
        # """Creates the layer from its config."""

        module_list_str = config.pop("moduleList")

        if OUTPUT_VARIABLES:
            print("module_list_str = ", module_list_str)

        moduleList = nn_Sequential.from_config( module_list_str )

        result =  cls(moduleList = moduleList, **config)

        if OUTPUT_POSITIONS:
            print("finish MLP: from_config")

        return result

    
    def call(self, x
             ):
        append = None

        if OUTPUT_FUNCTION_HEADER:
            print("mlp.py: MLP.call()")

        x = self.moduleList(x)

        return x





@register_keras_serializable(package="Custom")
class ResidualMLP(
    tf.keras.layers.Layer
    ):

    count_residualMLP = 0

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        my_layers = None,
        name = "ResidualMLP",
        **kwargs
    ):

        self.dim_list = list(dim_list)
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.use_layernorm_final = use_layernorm_final
        self.dropout = dropout


        if OUTPUT_FUNCTION_HEADER:
            print("mlp.py: ResidualMLP.__init__()", flush = True)

        super(ResidualMLP, self).__init__(name=name, **kwargs)

        hidden_dim = dim_list[1]

        num_hidden_layers = len(dim_list) - 3
        print("num_hidden_layers = ", num_hidden_layers)

        assert num_hidden_layers % 2 == 0

        if my_layers == None:

            if OUTPUT_VARIABLES:
                print("num_hidden_layers = ", num_hidden_layers)

            self.my_layers = [nn_Linear(dim_list[0], hidden_dim, name_Dense="ResidualMLP_my_layers_1" )]


            self.my_layers.extend(
                [
                    TwoLayerPreActivationResNetLinear(
                        hidden_dim=hidden_dim,
                        activation_type=activation_type,
                        use_layernorm=use_layernorm,
                        dropout=dropout,
                        l1_name="TwoLayerPreActivationResNetLinear_l1-" + str(_),
                        l2_name="TwoLayerPreActivationResNetLinear_l2-" + str(_),
                        name = "TwoLayerPreActivationResNetLinear-" + str(_)
                    )
                    for _ in range(1, num_hidden_layers, 2)
                ]
            )
            self.my_layers.append(nn_Linear(hidden_dim, dim_list[-1], name_Dense="ResidualMLP_my_layers_2" ))


            if use_layernorm_final:
                self.my_layers.append(nn_LayerNorm(dim_list[-1], name="nn_LayerNorm1"))
            self.my_layers.append(activation_dict[out_activation_type])
            
            self.my_layers = nn_Sequential(self.my_layers, name = "my_layers")

        else:
            self.my_layers = my_layers

    def get_config(self):
        config = super(ResidualMLP, self).get_config()  # Call the base class's get_config

        if OUTPUT_POSITIONS:
            # print every property with its type and value
            print("Checking ResidualMLP Config elements:")

        if OUTPUT_VARIABLES:
            print(f"dim_list: {self.dim_list}, type: {type(self.dim_list)}")
            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
            print(f"out_activation_type: {self.out_activation_type}, type: {type(self.out_activation_type)}")
            print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
            print(f"use_layernorm_final: {self.use_layernorm_final}, type: {type(self.use_layernorm_final)}")
            print(f"dropout: {self.dropout}, type: {type(self.dropout)}")

            print(f"ResidualMLP: name: {self.name}, type: {type(self.name)}")
        

        config.update({
            "dim_list": self.dim_list,  # Store dim_list (using input_shape to capture dims)
            "activation_type": self.activation_type,
            "out_activation_type": self.out_activation_type,
            "use_layernorm": self.use_layernorm,
            "use_layernorm_final": self.use_layernorm_final,
            "dropout": self.dropout,

            "my_layers": self.my_layers.get_config(),

            "name": "ResidualMLP",
            
        })
        return config
    

    @classmethod
    def from_config(cls, config):
        # """Creates the layer from its config."""

        my_layers = nn_Sequential.from_config(config.pop("my_layers"))

        config['name'] == "ResidualMLP"

        result =  cls(my_layers = my_layers, **config)

        if OUTPUT_POSITIONS:
            print("finish ResidualMLP: from_config")

        return result

    


    def call(self, x):

        if OUTPUT_FUNCTION_HEADER:
            print("mlp.py: ResidualMLP.call()", flush = True)

        x = self.my_layers(x)

        return x





class TwoLayerPreActivationResNetLinear(models.Model):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
        l1 = None,
        l2 = None,
        act = None,
        norm1 = None,
        norm2 = None,
        name="TwoLayerPreActivationResNetLinear",
        l1_name = None,
        l2_name = None,
        **kwargs
    ):
        self.hidden_dim = hidden_dim
        self.activation_type=activation_type
        self.use_layernorm = use_layernorm
        self.dropout = dropout

        if OUTPUT_FUNCTION_HEADER:
            print("mlp.py: TwoLayerPreActivationResNetLinear.__init__()", flush = True)

        super(TwoLayerPreActivationResNetLinear, self).__init__(name=name, **kwargs)
        if l1 == None:
            self.l1 = nn_Linear(hidden_dim, hidden_dim, name_Dense=l1_name)
        else:
            self.l1 = l1

        if l2 == None:
            self.l2 = nn_Linear(hidden_dim, hidden_dim, name_Dense=l2_name)
        else:
            self.l2 = l2
        
        if act == None:
            self.act = activation_dict[activation_type]
        else:
            self.act = act

        if use_layernorm:
            if norm1 == None:
                self.norm1 = nn_LayerNorm(hidden_dim, eps=1e-06)
            else:
                self.norm1 = norm1
    
            if norm2 == None:
                self.norm2 = nn_LayerNorm(hidden_dim, eps=1e-06)
            else:
                self.norm2 = norm2
                
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def call(self, x):

        if OUTPUT_FUNCTION_HEADER:
            print("mlp.py: TwoLayerPreActivationResNetLinear.forward()", flush = True)

        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))

        if OUTPUT_POSITIONS:
            print("mlp.py: TwoLayerPreActivationResNetLinear.forward() finished", flush = True)


        return x + x_input




    def get_config(self):
        config = super(TwoLayerPreActivationResNetLinear, self).get_config()  # Call the base class's get_config

        # print every property with its type and value
        if OUTPUT_POSITIONS:
            print("Checking TwoLayerPreActivationResNetLinear Config elements:")

        if OUTPUT_VARIABLES:
            print(f"hidden_dim: {self.hidden_dim}, type: {type(self.hidden_dim)}")
            print(f"activation_type: {self.activation_type}, type: {type(self.activation_type)}")
            print(f"use_layernorm: {self.use_layernorm}, type: {type(self.use_layernorm)}")
            print(f"dropout: {self.dropout}, type: {type(self.dropout)}")
        

        config.update({
            "hidden_dim": self.hidden_dim, 
            "activation_type": self.activation_type,
            "use_layernorm": self.use_layernorm,
            "dropout": self.dropout,

            "act": self.act.get_config(),

            "name": self.name,

        })


        if self.use_layernorm:
            config.update({
                "norm1": self.norm1.get_config(),
                "norm2": self.norm2.get_config(),
            })
        else:
            config.update({
                "norm1": None,
                "norm2": None,
            })


        if OUTPUT_VARIABLES:
            print("TwoLayerPreActivationResNetLinear config = ", config)
        
        return config
    

    @classmethod
    def from_config(cls, config):

        
        activation_type = config.pop("activation_type")
        
        act = activation_dict[activation_type]
        
        act = act.from_config( config.pop("act") )

        config_norm1 = config.pop("norm1")
        if config_norm1:
            norm1 = nn_LayerNorm.from_config( config_norm1 )
        else:
            norm1 = None

        config_norm2 = config.pop("norm2")
        if config_norm2:
            norm2 = nn_LayerNorm.from_config( config_norm2 )
        else:
            norm2 = None

        result = cls(
            act=act, norm1=norm1, norm2=norm2, activation_type=activation_type, **config)
        return result


