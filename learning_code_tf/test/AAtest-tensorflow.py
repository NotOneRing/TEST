import tensorflow as tf
print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


print(tf.sysconfig.get_build_info())










































