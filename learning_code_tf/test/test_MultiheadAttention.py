import tensorflow as tf
import torch
import numpy as np

# # 定义 TensorFlow 的 nn_MultiheadAttention 类
# class nn_MultiheadAttention(tf.keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, dropout=0.0):
#         super(nn_MultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout

#         # 使用 TensorFlow 的 MultiHeadAttention
#         self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
#         self.dropout_layer = tf.keras.layers.Dropout(dropout)

#     def call(self, query, key, value, mask=None):
#         """
#         query, key, value: [batch_size, seq_len, embed_dim]
#         mask: [batch_size, seq_len] or [batch_size, 1, seq_len, seq_len] (optional)
#         """
#         # 计算多头注意力
#         attention_output = self.attention(query=query, value=value, key=key, attention_mask=mask)
#         # 对输出应用 Dropout
#         output = self.dropout_layer(attention_output)
#         return output

# # 测试代码
# if __name__ == "__main__":
#     # 设置随机种子
#     torch.manual_seed(42)
#     tf.random.set_seed(42)

#     # 参数设置
#     embed_dim = 8
#     num_heads = 2
#     batch_size = 2
#     seq_len = 4
#     dropout = 0.1

#     # 随机生成输入张量
#     query = np.random.rand(batch_size, seq_len, embed_dim).astype(np.float32)
#     key = np.random.rand(batch_size, seq_len, embed_dim).astype(np.float32)
#     value = np.random.rand(batch_size, seq_len, embed_dim).astype(np.float32)

#     # PyTorch 部分
#     torch_attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
#     torch_query = torch.tensor(query)
#     torch_key = torch.tensor(key)
#     torch_value = torch.tensor(value)
#     torch_output, _ = torch_attention(torch_query, torch_key, torch_value)

#     # TensorFlow 部分
#     tf_attention = nn_MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
#     tf_query = tf.convert_to_tensor(query)
#     tf_key = tf.convert_to_tensor(key)
#     tf_value = tf.convert_to_tensor(value)
#     tf_output = tf_attention(tf_query, tf_key, tf_value)

#     # 打印输入和输出结果
#     print("Input Query:")
#     print(query)
#     print("\nPyTorch Output:")
#     print(torch_output.detach().numpy())
#     print("\nTensorFlow Output:")
#     print(tf_output.numpy())

#     print("np.allclose(torch_output.detach().numpy(), tf_output.numpy()) = ", np.allclose(torch_output.detach().numpy(), tf_output.numpy()))

#     print("torch_output.detach().numpy() - tf_output.numpy() = ", torch_output.detach().numpy() - tf_output.numpy())









from util.torch_to_tf import nn_MultiheadAttention

import torch
import tensorflow as tf
import numpy as np

# 设置随机种子
torch.manual_seed(42)
batch_size, seq_len, d_model = 2, 5, 16

# 在 PyTorch 中生成随机数据
query_torch = torch.randn(batch_size, seq_len, d_model)
key_torch = torch.randn(batch_size, seq_len, d_model)
value_torch = torch.randn(batch_size, seq_len, d_model)

# 转换为 TensorFlow 张量
query_tf = tf.convert_to_tensor(query_torch.numpy())
key_tf = tf.convert_to_tensor(key_torch.numpy())
value_tf = tf.convert_to_tensor(value_torch.numpy())




import torch.nn as nn

# 创建 PyTorch MultiheadAttention
num_heads = 4
attention_torch = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

# 获取 PyTorch 参数
query_weight_torch = attention_torch.in_proj_weight
query_bias_torch = attention_torch.in_proj_bias
output_weight_torch = attention_torch.out_proj.weight
output_bias_torch = attention_torch.out_proj.bias



# 初始化 TensorFlow MultiheadAttention
attention_tf = nn_MultiheadAttention(num_heads=num_heads, d_model=d_model)



# 使用虚拟输入触发权重初始化
dummy_query = tf.random.normal((1, 1, d_model))
dummy_key = tf.random.normal((1, 1, d_model))
dummy_value = tf.random.normal((1, 1, d_model))
attention_tf(dummy_query, dummy_key, dummy_value)


# 设置参数
# 设置参数
attention_tf.query_dense.kernel.assign(
    tf.convert_to_tensor(query_weight_torch[:d_model].detach().numpy().T)
)
attention_tf.query_dense.bias.assign(
    tf.convert_to_tensor(query_bias_torch[:d_model].detach().numpy())
)

attention_tf.key_dense.kernel.assign(
    tf.convert_to_tensor(query_weight_torch[d_model:2*d_model].detach().numpy().T)
)
attention_tf.key_dense.bias.assign(
    tf.convert_to_tensor(query_bias_torch[d_model:2*d_model].detach().numpy())
)

attention_tf.value_dense.kernel.assign(
    tf.convert_to_tensor(query_weight_torch[2*d_model:].detach().numpy().T)
)
attention_tf.value_dense.bias.assign(
    tf.convert_to_tensor(query_bias_torch[2*d_model:].detach().numpy())
)

attention_tf.output_dense.kernel.assign(
    tf.convert_to_tensor(output_weight_torch.detach().numpy().T)
)
attention_tf.output_dense.bias.assign(
    tf.convert_to_tensor(output_bias_torch.detach().numpy())
)







# 执行 PyTorch MultiheadAttention
output_torch, attention_weights_torch = attention_torch(query_torch, key_torch, value_torch)

# 执行 TensorFlow MultiheadAttention
output_tf, attention_weights_tf = attention_tf(query_tf, key_tf, value_tf)


# attention_weights_tf = tf.reduce_mean(attention_weights_tf, axis=1)  # 平均所有头



# 转换为 NumPy
output_torch_np = output_torch.detach().numpy()
attention_weights_torch_np = attention_weights_torch.detach().numpy()

output_tf_np = output_tf.numpy()
attention_weights_tf_np = attention_weights_tf.numpy()


print("output_torch_np = ", output_torch_np)

print("output_tf_np = ", output_tf_np)

print("attention_weights_torch_np = ", attention_weights_torch_np)
print("attention_weights_tf_np = ", attention_weights_tf_np)


# 计算误差
output_diff = np.mean(np.abs(output_torch_np - output_tf_np))
attention_weights_diff = np.mean(np.abs(attention_weights_torch_np - attention_weights_tf_np))

print(f"Output difference: {output_diff}")
print(f"Attention weights difference: {attention_weights_diff}")












