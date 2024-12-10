import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[7, 8, 9], [10, 11, 12]])

# 使用 tf.concat 和 tf.stack
c1 = tf.concat([a, b], axis=0)
s1 = tf.stack([a, b], axis=0)
c2 = tf.concat([a, b], axis=1)
s2 = tf.stack([a, b], axis=1)


# 打印结果（直接打印张量）
print("c1 (concat axis=0):\n", c1.numpy())
print("\n")

print("s1 (stack axis=0):\n", s1.numpy())
print("\n")

print("c2 (concat axis=1):\n", c2.numpy())
print("\n")

print("s2 (stack axis=1):\n", s2.numpy())


# c3 = tf.concat([a, b])
# print("c2 (concat axis=None):\n", c3.numpy())
# print("\n")

s3 = tf.stack([a, b])



print("s3 (stack axis=None):\n", s3.numpy())

print("s3.shape (stack axis=None):\n", s3.numpy().shape)





































