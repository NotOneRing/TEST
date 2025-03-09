import numpy as np
import tensorflow as tf
import random


from util.torch_to_tf import torch_utils_data_DataLoader




# example dataset
dataset = [
    {
    "actions": np.random.rand(3), 
    "states": np.random.rand(4), 
    "rewards": np.random.rand(1),
    "next_states": np.random.rand(4), 
    "rgb": np.random.rand(224, 224, 3)}
    for _ in range(100)
]



# create DataLoader
dataloader = torch_utils_data_DataLoader(dataset, batch_size=5, shuffle=True)



# iterate data
for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    for key, value in batch.items():
        print(f"  {key}: shape={value.shape}")


























