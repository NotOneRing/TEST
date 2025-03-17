import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

# set random seeds to ensure the reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)



# create a TransformerDecoderLayer
d_model = 4
nhead = 2
decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=16, dropout=0)

decoder_layer.eval()

# state_dict = decoder_layer.state_dict()
# for name, param in state_dict.items():
#     print(name, param.shape)


# input data with batch=1
torch_tgt = torch.tensor([[[0.1, 0.2, 0.3, 0.4]],
                    [[0.5, 0.6, 0.7, 0.8]]])  # (tgt_len=2, batch_size=1, d_model=4)

torch_memory = torch.tensor([[[0.9, 1.0, 1.1, 1.2]],
                       [[1.3, 1.4, 1.5, 1.6]]])  # (memory_len=2, batch_size=1, d_model=4)

# forward pass
torch_output = decoder_layer(torch_tgt, torch_memory)

# print output
print("TransformerDecoderLayer output: \n", torch_output)

print("TransformerDecoderLayer output.shape: \n", torch_output.shape)


from util.torch_to_tf import nn_TransformerDecoderLayer, torch_tensor

# create a TransformerDecoderLayer
d_model = 4
nhead = 2
tf_decoder_layer = nn_TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=16, dropout=0)


# input data with batch=1
tf_tgt = torch_tensor( np.array([[[0.1, 0.2, 0.3, 0.4]],
                    [[0.5, 0.6, 0.7, 0.8]]]) )  # (tgt_len=2, batch_size=1, d_model=4)

tf_memory = torch_tensor( np.array([[[0.9, 1.0, 1.1, 1.2]],
                       [[1.3, 1.4, 1.5, 1.6]]]) )  # (memory_len=2, batch_size=1, d_model=4)

# forward pass
tf_output = tf_decoder_layer(tf_tgt, tf_memory)


self_attn_query_weight_torch = decoder_layer.self_attn.in_proj_weight
self_attn_query_bias_torch = decoder_layer.self_attn.in_proj_bias
self_attn_output_weight_torch = decoder_layer.self_attn.out_proj.weight
self_attn_output_bias_torch = decoder_layer.self_attn.out_proj.bias


dummy_query = tf.random.normal((1, 1, d_model))
dummy_key = tf.random.normal((1, 1, d_model))
dummy_value = tf.random.normal((1, 1, d_model))
tf_decoder_layer.self_attn(dummy_query, dummy_key, dummy_value)



# Set parameters from PyTorch to TensorFlow
tf_decoder_layer.self_attn.query_dense.kernel.assign(
    tf.convert_to_tensor(decoder_layer.self_attn.in_proj_weight[:d_model].detach().numpy().T)
)
tf_decoder_layer.self_attn.query_dense.bias.assign(
    tf.convert_to_tensor(decoder_layer.self_attn.in_proj_bias[:d_model].detach().numpy())
)
tf_decoder_layer.self_attn.key_dense.kernel.assign(
    tf.convert_to_tensor(decoder_layer.self_attn.in_proj_weight[d_model:2*d_model].detach().numpy().T)
)
tf_decoder_layer.self_attn.key_dense.bias.assign(
    tf.convert_to_tensor(decoder_layer.self_attn.in_proj_bias[d_model:2*d_model].detach().numpy())
)
tf_decoder_layer.self_attn.value_dense.kernel.assign(
    tf.convert_to_tensor(decoder_layer.self_attn.in_proj_weight[2*d_model:].detach().numpy().T)
)
tf_decoder_layer.self_attn.value_dense.bias.assign(
    tf.convert_to_tensor(decoder_layer.self_attn.in_proj_bias[2*d_model:].detach().numpy())
)
tf_decoder_layer.self_attn.output_dense.kernel.assign(
    tf.convert_to_tensor( decoder_layer.self_attn.out_proj.weight.detach().numpy().T )
)
tf_decoder_layer.self_attn.output_dense.bias.assign(
    tf.convert_to_tensor( decoder_layer.self_attn.out_proj.bias.detach().numpy() )
)






self_multihead_attn_query_weight_torch = decoder_layer.self_attn.in_proj_weight
self_multihead_attn_query_bias_torch = decoder_layer.self_attn.in_proj_bias
self_multihead_attn_output_weight_torch = decoder_layer.self_attn.out_proj.weight
self_multihead_attn_output_bias_torch = decoder_layer.self_attn.out_proj.bias


dummy_query = tf.random.normal((1, 1, d_model))
dummy_key = tf.random.normal((1, 1, d_model))
dummy_value = tf.random.normal((1, 1, d_model))
tf_decoder_layer.multihead_attn(dummy_query, dummy_key, dummy_value)


tf_decoder_layer.multihead_attn.query_dense.kernel.assign(
    tf.convert_to_tensor(decoder_layer.multihead_attn.in_proj_weight[:d_model].detach().numpy().T)
)
tf_decoder_layer.multihead_attn.query_dense.bias.assign(
    tf.convert_to_tensor(decoder_layer.multihead_attn.in_proj_bias[:d_model].detach().numpy())
)
tf_decoder_layer.multihead_attn.key_dense.kernel.assign(
    tf.convert_to_tensor(decoder_layer.multihead_attn.in_proj_weight[d_model:2*d_model].detach().numpy().T)
)
tf_decoder_layer.multihead_attn.key_dense.bias.assign(
    tf.convert_to_tensor(decoder_layer.multihead_attn.in_proj_bias[d_model:2*d_model].detach().numpy())
)
tf_decoder_layer.multihead_attn.value_dense.kernel.assign(
    tf.convert_to_tensor(decoder_layer.multihead_attn.in_proj_weight[2*d_model:].detach().numpy().T)
)
tf_decoder_layer.multihead_attn.value_dense.bias.assign(
    tf.convert_to_tensor(decoder_layer.multihead_attn.in_proj_bias[2*d_model:].detach().numpy())
)
tf_decoder_layer.multihead_attn.output_dense.kernel.assign(
    tf.convert_to_tensor( decoder_layer.multihead_attn.out_proj.weight.detach().numpy().T )
)
tf_decoder_layer.multihead_attn.output_dense.bias.assign(
    tf.convert_to_tensor( decoder_layer.multihead_attn.out_proj.bias.detach().numpy() )
)



tf_decoder_layer.linear1.trainable_weights[0].assign( decoder_layer.linear1.weight.detach().numpy().T )
tf_decoder_layer.linear1.trainable_weights[1].assign( decoder_layer.linear1.bias.detach().numpy() )
tf_decoder_layer.linear2.trainable_weights[0].assign( decoder_layer.linear2.weight.detach().numpy().T )
tf_decoder_layer.linear2.trainable_weights[1].assign( decoder_layer.linear2.bias.detach().numpy() )

tf_decoder_layer.norm1.trainable_weights[0].assign( decoder_layer.norm1.weight.detach().numpy() )
tf_decoder_layer.norm1.trainable_weights[1].assign( decoder_layer.norm1.bias.detach().numpy() )
tf_decoder_layer.norm2.trainable_weights[0].assign( decoder_layer.norm2.weight.detach().numpy() )
tf_decoder_layer.norm2.trainable_weights[1].assign( decoder_layer.norm2.bias.detach().numpy() )
tf_decoder_layer.norm3.trainable_weights[0].assign( decoder_layer.norm3.weight.detach().numpy() )
tf_decoder_layer.norm3.trainable_weights[1].assign( decoder_layer.norm3.bias.detach().numpy() )



tf_output = tf_decoder_layer(tf_tgt, tf_memory, training=False)

# print output
print("TransformerDecoderLayer output: \n", tf_output)



tgt_mask=None
memory_mask=None
tgt_key_padding_mask=None
memory_key_padding_mask=None



state_dict = decoder_layer.self_attn.state_dict()
for name, param in state_dict.items():
    print(name, param.shape)


# 1. Self-Attention (Decoder)
if tf_decoder_layer.norm_first:
    tf_tgt2 = tf_decoder_layer.self_attn(tf_decoder_layer.norm1(tf_tgt), tf_decoder_layer.norm1(tf_tgt), tf_decoder_layer.norm1(tf_tgt),
                            attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

    torch_tgt2 = decoder_layer.self_attn(decoder_layer.norm1(torch_tgt), decoder_layer.norm1(torch_tgt), decoder_layer.norm1(torch_tgt),
                            attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]


    print("tf_tgt2 = ", tf_tgt2)
    print("torch_tgt2 = ", torch_tgt2)

    assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-3 )

    # if training:
    tf_tgt = tf_tgt + tf_decoder_layer.dropout1(tf_tgt2)

    torch_tgt = torch_tgt + tf_decoder_layer.dropout1(torch_tgt2)

    assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-3 )

    # else:
    #     tgt = tgt + tgt2
else:
    tf_tgt2 = tf_decoder_layer.self_attn(tf_tgt, tf_tgt, tf_tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
    # if training:

    torch_tgt2 = decoder_layer.self_attn(torch_tgt, torch_tgt, torch_tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

    print("tf_tgt2.numpy() = ", tf_tgt2.numpy())
    print("torch_tgt2.detach().numpy() = ", torch_tgt2.detach().numpy())

    assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-3 )

    tf_tgt = tf.cast( tf_tgt, tf.float32 )
    tf_tgt2 = tf.cast( tf_tgt2, tf.float32 )

    tf_tgt = tf_decoder_layer.norm1(tf_tgt + tf_decoder_layer.dropout1(tf_tgt2))


    torch_tgt = decoder_layer.norm1(torch_tgt + decoder_layer.dropout1(torch_tgt2))

    # else:
    #     tgt = self.norm1(tgt + tgt2)

    assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-3 )


# 2. Cross-Attention (Encoder-Decoder)
if tf_decoder_layer.norm_first:
    tf_tgt2 = tf_decoder_layer.multihead_attn(tf_decoder_layer.norm2(tf_tgt), tf_decoder_layer.norm2(tf_memory), tf_decoder_layer.norm2(tf_memory),
                                attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

    torch_tgt2 = decoder_layer.multihead_attn(decoder_layer.norm2(torch_tgt), decoder_layer.norm2(torch_memory), decoder_layer.norm2(torch_memory),
                                attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]



    tf_tgt2 = tf.convert_to_tensor( torch_tgt2.detach().numpy().astype(np.float32) )
    tf_tgt = tf.convert_to_tensor( torch_tgt.detach().numpy().astype(np.float32) )


    # if training:
    tf_tgt = tf_tgt + tf_decoder_layer.dropout2(tf_tgt2)

    torch_tgt = torch_tgt + decoder_layer.dropout2(torch_tgt2)
    # else:
    #     tgt = tgt + tgt2
else:
    tf_tgt2 = tf_decoder_layer.multihead_attn(tf_tgt, tf_memory, tf_memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

    torch_tgt2 = decoder_layer.multihead_attn(torch_tgt, torch_memory, torch_memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

    # # if training:
    # result1 = tgt + tf_decoder_layer.dropout2(tgt2)
    # result2 = tgt + tgt2
    # print("tgt2 = ", tgt2)
    # print("tf_decoder_layer.dropout2(tgt2) = ", tf_decoder_layer.dropout2(tgt2))
    # assert np.allclose( result1.numpy(), result2.numpy(), atol=1e-4), "not equal"
 

    tf_tgt2 = tf.convert_to_tensor( torch_tgt2.detach().numpy().astype(np.float32) )
    tf_tgt = tf.convert_to_tensor( torch_tgt.detach().numpy().astype(np.float32) )

 
    tf_tgt = tf_decoder_layer.norm2(tf_tgt + tf_decoder_layer.dropout2(tf_tgt2))

    torch_tgt = decoder_layer.norm2(torch_tgt + decoder_layer.dropout2(torch_tgt2))


# 3. Feedforward Network (FFN)
if tf_decoder_layer.norm_first:
    # if training:
    tf_tgt2 = tf_decoder_layer.linear2(tf_decoder_layer.dropout(tf_decoder_layer.activation(tf_decoder_layer.linear1(tf_decoder_layer.norm3(tf_tgt)))))
    tf_tgt = tf_tgt + tf_decoder_layer.dropout3(tf_tgt2)


    torch_tgt2 = decoder_layer.linear2( decoder_layer.dropout( decoder_layer.activation( decoder_layer.linear1( decoder_layer.norm3( torch_tgt ) ) ) ) )
    torch_tgt = torch_tgt + decoder_layer.dropout3(torch_tgt2)

    # else:
    #     tgt2 = self.linear2( self.activation(self.linear1(self.norm3(tgt))) )
    #     tgt = tgt + tgt2
else:
    # if training:
    tf_tgt2 = tf_decoder_layer.linear2(tf_decoder_layer.dropout(tf_decoder_layer.activation(tf_decoder_layer.linear1(tf_tgt))))
    tf_tgt = tf_decoder_layer.norm3(tf_tgt + tf_decoder_layer.dropout3(tf_tgt2))

    torch_tgt2 = decoder_layer.linear2( decoder_layer.dropout( decoder_layer.activation( decoder_layer.linear1(torch_tgt) ) ) )
    torch_tgt = decoder_layer.norm3( torch_tgt + decoder_layer.dropout3( torch_tgt2 ) )


assert np.allclose( tf_tgt.numpy(), torch_tgt.detach().numpy() , atol=1e-4 )
assert np.allclose( tf_tgt2.numpy(), torch_tgt2.detach().numpy() , atol=1e-4 )


assert np.allclose( tf_tgt.numpy(), torch_output.detach().numpy() , atol=1e-4 )



    # else:
    #     tgt2 = self.linear2(self.activation(self.linear1(tgt)))
    #     tgt = tgt + tgt2
















