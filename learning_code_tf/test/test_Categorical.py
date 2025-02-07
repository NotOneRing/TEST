import torch

# def probs_to_logits(probs):
#     # 为了避免计算 log(0)，首先加上一个小的常数 eps
#     eps = 1e-8
#     probs = torch.clamp(probs, min=eps, max=1 - eps)  # 防止数值不稳定
#     logits = torch.log(probs / (1 - probs))  # 根据logit公式进行转换
#     return logits

# logits = torch.tensor([0.5, 1.0, -0.5])

# probs = torch.softmax(logits, dim=-1)


# print("logits = ", logits)

# print("probs = ", probs)

# print("probs_to_logits(probs) = ", probs_to_logits(probs))

class CategoricalDistribution:
    def __init__(self, probs=None, logits=None):
        if probs is not None:
            self.probs = probs

        elif logits is not None:
            self.logits = logits
            self.probs = torch.softmax(logits, dim=-1)
        else:
            raise ValueError("Must specify either probs or logits.")
    
    def sample(self):
        return torch.multinomial(self.probs, num_samples=1)

        # return tf.random.categorical(self.probs, num_samples = 1, dtype=tf.int32)

    def log_prob(self, value):
        # assert len(value.shape.as_list()) <= 2
        if self.probs is not None:

            print("torch.log(self.probs) = ", torch.log(self.probs) )

            print("value = ", value)

            value_shape_list = list(value.shape)

            print("value_shape_list = ", value_shape_list)

            batch_dim = value_shape_list[0]

            all_tensors = []

            for i in range(batch_dim):
                all_tensors.append( torch.log( self.probs[..., value[i, ...]] ).reshape(1, -1) )

            if batch_dim == 1:
                result = all_tensors[0]
            else:
                result = torch.cat(all_tensors, dim=0)

            return result
            # return torch.log(self.probs[..., value])
            # return torch.gather(self.probs[..., value])

        else:  # logits provided
            # return self.logits[..., value] - torch.logsumexp(self.logits, dim=-1)
            raise ValueError("Must specify probs.")


    def entropy(self):
        return -torch.sum(self.probs * torch.log(self.probs), dim=-1)

# 示例用法
probs = torch.tensor([0.1, 0.7, 0.2])
dist = CategoricalDistribution(probs=probs)

# 采样
sample = dist.sample()
print(f"Sampled value: {sample}")

x = torch.tensor([1])
# 对数概率
log_prob = dist.log_prob(x)  # 计算类别 1 的对数概率
print(f"Log probability of class 1: {log_prob}")

# 熵
entropy = dist.entropy()
print(f"Entropy: {entropy}")


from torch.distributions.categorical import Categorical as tcc

import torch

# 创建批量的 logits
logits_batch = torch.tensor([[1.0, 0.5, -0.5], [0.2, 0.7, 0.1]])

# 创建批量分类分布对象
dist_batch = CategoricalDistribution(logits=logits_batch)

# 采样批量样本
samples_batch = dist_batch.sample()
print(f"1Sampled indices from batch: {samples_batch}")

# 计算批量样本的对数概率
log_prob_batch = dist_batch.log_prob(samples_batch)
print(f"1Log probabilities of sampled indices: {log_prob_batch}")

print("log_prob_batch.shape = ", log_prob_batch.shape)

# 计算批量的熵
entropy_batch = dist_batch.entropy()
print(f"1Entropy of batch: {entropy_batch}")



# 创建批量分类分布对象
dist_batch = tcc(logits=logits_batch)

# # 采样批量样本
# samples_batch = dist_batch.sample()
print(f"2Sampled indices from batch: {samples_batch}")

# 计算批量样本的对数概率
log_prob_batch = dist_batch.log_prob(samples_batch)
print(f"2Log probabilities of sampled indices: {log_prob_batch}")
print("log_prob_batch.shape = ", log_prob_batch.shape)

# 计算批量的熵
entropy_batch = dist_batch.entropy()
print(f"2Entropy of batch: {entropy_batch}")



# a = torch.Size([])
# print("a.numel() = ", a.numel())





from util.torch_to_tf import Categorical


import tensorflow as tf


# 创建批量分类分布对象
dist_batch = Categorical(logits= tf.convert_to_tensor(logits_batch.numpy()) ) 

samples_batch = dist_batch.sample()

# # 采样批量样本
# samples_batch = dist_batch.sample()
print(f"3Sampled indices from batch: {samples_batch}")

# 计算批量样本的对数概率
log_prob_batch = dist_batch.log_prob(samples_batch)
print(f"3Log probabilities of sampled indices: {log_prob_batch}")
print("log_prob_batch.shape = ", log_prob_batch.shape)

# 计算批量的熵
entropy_batch = dist_batch.entropy()
print(f"3Entropy of batch: {entropy_batch}")




# dist = tcc(probs = probs)

# # 采样
# sample = dist.sample()
# print(f"Sampled value: {sample}")

# # 对数概率
# log_prob = dist.log_prob( x )  # 计算类别 1 的对数概率
# print(f"Log probability of class 1: {log_prob}")

# # 熵
# entropy = dist.entropy()
# print(f"Entropy: {entropy}")




# import torch

# # 创建logits
# logits = torch.tensor([0.5, 1.0, -0.5])

# # 创建CategoricalDistribution对象
# dist = CategoricalDistribution(logits=logits)

# # 采样一个样本
# sample = dist.sample()
# print(f"Sampled index: {sample.item()}")

# # 计算样本的对数概率
# log_prob = dist.log_prob(sample)
# print(f"Log probability of sampled index {sample.item()}: {log_prob.item()}")

# # 计算熵
# entropy = dist.entropy()
# print(f"Entropy: {entropy.item()}")


# dist2 = tcc(logits=logits)
# # 采样一个样本
# sample = dist.sample()
# print(f"Sampled index: {sample.item()}")

# # 计算样本的对数概率
# log_prob = dist.log_prob(sample)
# print(f"Log probability of sampled index {sample.item()}: {log_prob.item()}")

# # 计算熵
# entropy = dist.entropy()
# print(f"Entropy: {entropy.item()}")










# import torch

# # 使用logits创建分布
# logits = torch.tensor([1.0, 0.5, -0.5])
# dist_from_logits = CategoricalDistribution(logits=logits)

# # 使用probs创建相同的分布
# probs = torch.softmax(logits, dim=-1)  # 转换logits为probs
# dist_from_probs = CategoricalDistribution(probs=probs)

# # 比较两个分布的采样结果
# sample_logits = dist_from_logits.sample()
# sample_probs = dist_from_probs.sample()

# print(f"Sample from logits-based distribution: {sample_logits.item()}")
# print(f"Sample from probs-based distribution: {sample_probs.item()}")

# # 比较两个分布的对数概率
# log_prob_logits = dist_from_logits.log_prob(sample_logits)
# log_prob_probs = dist_from_probs.log_prob(sample_probs)

# print(f"Log probability (logits-based): {log_prob_logits.item()}")
# print(f"Log probability (probs-based): {log_prob_probs.item()}")



# dist_from_logits = tcc(logits=logits)

# dist_from_probs = tcc(probs=probs)

# # 比较两个分布的采样结果
# sample_logits = dist_from_logits.sample()
# sample_probs = dist_from_probs.sample()

# print(f"Sample from logits-based distribution: {sample_logits.item()}")
# print(f"Sample from probs-based distribution: {sample_probs.item()}")

# # 比较两个分布的对数概率
# log_prob_logits = dist_from_logits.log_prob(sample_logits)
# log_prob_probs = dist_from_probs.log_prob(sample_probs)

# print(f"Log probability (logits-based): {log_prob_logits.item()}")
# print(f"Log probability (probs-based): {log_prob_probs.item()}")





















































