import tensorflow as tf

# define an example tensor
# tensor = tf.constant([[1, 2, 3], 
#                       [4, 5, 6], 
#                       [7, 8, 9]])
tensor = tf.range(0, 27, 1)

tensor = tf.reshape(tensor, (3, 3, 3))

print("tensor = ", tensor)

# define indices list
indices = tf.constant([[0, 0], [1, 1], [2, 2]])

# use tf.gather_nd
result = tf.gather_nd(tensor, indices)

print(result.numpy())  # output [1, 5, 9]



# suppose we need index along dim 0 and dim 1 simultaneously
rows = tf.constant([0, 1, 2])  # index along 0 dim
cols = tf.constant([0, 1, 2])  # index along 1 dim

# use tf.stack to create indices
indices = tf.stack([rows, cols], axis=1)

# use tf.gather_nd to get values
result = tf.gather_nd(tensor, indices)

print(result.numpy())  # output [1, 5, 9]


