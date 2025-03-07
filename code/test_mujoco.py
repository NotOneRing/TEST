


import tensorflow_datasets as tfds

# # load d4rl_mujoco_hopper/v2-medium dataset
# dataset, info = tfds.load('d4rl_mujoco_hopper/v2-medium', with_info=True, as_supervised=True)

# # View dataset information
# print(info)

# # Get the training set
# train_data = dataset['train']

# # Preview a portion of the dataset
# for example in train_data.take(5):
#     states, actions = example
#     print('States:', states.numpy())
#     print('Actions:', actions.numpy())


dataset = tfds.load('d4rl_mujoco_hopper/v2-medium', as_supervised=False)
# Get dataset information
# print(info)

# View the structure of the dataset
for example in dataset['train']:
    print(example)
    break





