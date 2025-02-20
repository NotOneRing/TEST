


import tensorflow_datasets as tfds

# # 加载 d4rl_mujoco_hopper/v2-medium 数据集
# dataset, info = tfds.load('d4rl_mujoco_hopper/v2-medium', with_info=True, as_supervised=True)

# # 查看数据集的信息
# print(info)

# # 获取训练集
# train_data = dataset['train']

# # 预览数据集中的一部分
# for example in train_data.take(5):
#     states, actions = example
#     print('States:', states.numpy())
#     print('Actions:', actions.numpy())


dataset = tfds.load('d4rl_mujoco_hopper/v2-medium', as_supervised=False)
# 获取数据集中的信息
# print(info)

# 查看数据集内容结构
for example in dataset['train']:
    print(example)
    break





