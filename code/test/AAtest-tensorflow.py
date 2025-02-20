import tensorflow as tf
print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


print(tf.sysconfig.get_build_info())




print("CUDA version:", tf.sysconfig.get_build_info()['cuda_version'])
print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])








a = tf.reshape( tf.linspace(0, 20, 20), (2, 2, 5) )


print("a = ", a)


b = tf.range(start = 0, limit=5, delta=2)


print("b = ", b)


c = tf.gather(a, b, axis=2)



print("c = ", c)

























