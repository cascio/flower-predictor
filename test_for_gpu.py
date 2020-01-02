import tensorflow as tf

tf.debugging.set_log_device_placement(True)

is_built_with_cuda = tf.test.is_built_with_cuda()
print(is_built_with_cuda)

is_gpu_available = tf.test.is_gpu_available()
print(is_gpu_available)

version = tf.__version__
print(version)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # noqa


# Create some tensors to see that they're handled by the GPU
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)
