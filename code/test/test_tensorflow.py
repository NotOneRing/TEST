import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Dense


# import tensorflow.keras.Model as Model

model_1 = tf.keras.applications.VGG16(weights = 'imagenet',include_top = False)
model_1.trainable = False
model_1.summary()
layers = [l for l in model_1.layers]

for layer in layers:
  layer._outbound_nodes = []


input = keras.Input(shape=(224, 224, 3))
x = layers[1](input)
x = layers[2](x)
x = layers[3](x)
x = layers[4](x)
x = layers[5](x)
x = layers[6](x)
x = layers[7](x)
x = layers[8](x)
x = layers[9](x)
x = layers[10](x)
x = layers[11](x)
x = layers[12](x)
x = layers[13](x)
out = Dense(38, activation='softmax')(x)


result_model = tf.keras.Model(inputs=input, outputs=out)
result_model.summary()