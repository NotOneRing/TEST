import tensorflow as tf

# 创建简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,), activation='relu'),
    tf.keras.layers.Dense(2)
])

# 查看 trainable_variables
trainable_vars = model.trainable_variables

print("type(trainable_vars) = ", type(trainable_vars))



for var in trainable_vars:
    print("type(var) = ", type(var))

    print(f"Name: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}")



















