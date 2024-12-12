import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

from util.torch_to_tf import nn_ModuleList


# 初始化对齐函数
def align_initialization(torch_modulelist, tf_modulelist):
    for torch_layer, tf_layer in zip(torch_modulelist, tf_modulelist):
        if isinstance(torch_layer, nn.Linear) and isinstance(tf_layer, tf.keras.layers.Dense):

            # print("torch_layer.shape = ", torch_layer.weight.detach().shape)
            # print("tf_layer.shape = ", tf_layer.kernel.shape)

            dummy_input = tf.random.normal([1, torch_layer.in_features])
            tf_layer(dummy_input)  # Trigger weight initialization

            # 提取 PyTorch 初始化权重和偏置
            torch_weight = torch_layer.weight.detach().numpy()
            torch_bias = torch_layer.bias.detach().numpy() if torch_layer.bias is not None else None
            
            # 赋值给 TensorFlow 层
            tf_layer.kernel.assign(torch_weight.T)  # 转置权重
            if torch_bias is not None:
                tf_layer.bias.assign(torch_bias)
            else:
                print("torch_bias is None")



# PyTorch 测试函数
def test_torch_modulelist(module_list, input_tensor):


    outputs = []
    for module in module_list:
        input_tensor = module(input_tensor)
        outputs.append(input_tensor)

    return outputs



# TensorFlow 测试函数
def test_tf_modulelist(module_list, input_tensor):


    outputs = module_list(input_tensor)
    return outputs



# 比较测试
def compare_results():
    # 创建相同的输入张量
    torch_input = torch.randn(1, 10)
    tf_input = tf.convert_to_tensor(torch_input.detach().numpy(), dtype=tf.float32)
    
    # # 获取 PyTorch 和 TensorFlow 的模型及输出
    # torch_modulelist, torch_outputs = test_torch_modulelist(torch_input)
    # tf_modulelist, tf_outputs = test_tf_modulelist(tf_input)
    
    torch_modulelist = nn.ModuleList([
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ])

    tf_modulelist = nn_ModuleList([
        tf.keras.layers.Dense(20, activation=None),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(5, activation=None)
    ])

    # 对齐初始化
    align_initialization(torch_modulelist, tf_modulelist)
    

    # 获取 PyTorch 和 TensorFlow 的模型及输出
    torch_outputs = test_torch_modulelist(torch_modulelist, torch_input)
    tf_outputs = test_tf_modulelist(tf_modulelist, tf_input)
    

    # 确保输出层数一致
    assert len(torch_outputs) == len(tf_outputs), \
        f"Mismatch in number of layers: {len(torch_outputs)} (torch) vs {len(tf_outputs)} (tf)"
    
    # 比较每层输出
    for i, (torch_out, tf_out) in enumerate(zip(torch_outputs, tf_outputs)):
        # 转换为 NumPy 数组
        torch_out_np = torch_out.detach().numpy()
        tf_out_np = tf_out.numpy()
        
        # 检查形状是否一致
        if torch_out_np.shape != tf_out_np.shape:
            print(f"Layer {i} shape mismatch: Torch {torch_out_np.shape}, TF {tf_out_np.shape}")
        else:
            print(f"Layer {i} shape match: {torch_out_np.shape}")
        
        # 检查值是否接近
        match = np.allclose(torch_out_np, tf_out_np, atol=1e-5)
        print(f"Layer {i} output match: {match}")
        
        # 如果不匹配，打印差异
        if not match:
            diff = np.abs(torch_out_np - tf_out_np)
            print(f"Layer {i} max difference: {np.max(diff)}")
            print(f"Layer {i} difference matrix:\n{diff}")
        
        # 打印每层的输出值（可选）
        print(f"Layer {i} Torch output:\n{torch_out_np}")
        print(f"Layer {i} TF output:\n{tf_out_np}")
        print("-" * 50)



# 执行测试
compare_results()
