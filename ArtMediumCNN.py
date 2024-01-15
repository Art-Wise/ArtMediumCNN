import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from keras.layers import PReLU, Conv2D, BatchNormalization, MaxPooling2D, Lambda, Concatenate, Dense, Flatten, Dropout, Input, LeakyReLU
from keras.initializers import Constant

input = Input(shape=(230,230,5))

rgb_channel = Lambda(lambda x : x[:,:,:,:3])(input) # Use only rgb channels
gray_channel = Lambda(lambda x : x[:,:,:,3:4])(input) # Use only gray channel
edge_channel = Lambda(lambda x : x[:,:,:,4:5])(input) # Use only edge channel

# RGB
conv_1 = Conv2D(filters=64, kernel_size=(10,10), strides=(3,3))(rgb_channel)
conv_1 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(conv_1)
conv_1 = BatchNormalization()(conv_1)
conv_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(conv_1)

conv_2 = Conv2D(filters=96, kernel_size=(5,5), strides=(2,2))(conv_1)
conv_2 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(conv_2)

conv_3 = Conv2D(filters=96, kernel_size=(5,5), strides=(2,2))(conv_2)
conv_3 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(conv_3)
conv_3 = BatchNormalization()(conv_3)
conv_3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(conv_3)

#Gray
gray_conv_1 = Conv2D(filters=64, kernel_size=(10,10), strides=(3,3))(gray_channel)
gray_conv_1 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(gray_conv_1)
gray_conv_1 = BatchNormalization()(gray_conv_1)
gray_conv_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(gray_conv_1)
gray_conv_1 = Dropout(0.6)(gray_conv_1)

gray_conv_2 = Conv2D(filters=96, kernel_size=(5,5), strides=(2,2))(gray_conv_1)
gray_conv_2 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(gray_conv_2)

gray_conv_3 = Conv2D(filters=96, kernel_size=(5,5), strides=(2,2))(gray_conv_2)
gray_conv_3 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(gray_conv_3)
gray_conv_3 = BatchNormalization()(gray_conv_3)
gray_conv_3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(gray_conv_3)

#Edge
edge_conv_1 = Conv2D(filters=64, kernel_size=(10,10), strides=(3,3))(edge_channel)
edge_conv_1 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(edge_conv_1)
edge_conv_1 = BatchNormalization()(edge_conv_1)
edge_conv_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(edge_conv_1)
edge_conv_1 = Dropout(0.6)(edge_conv_1)

edge_conv_2 = Conv2D(filters=96, kernel_size=(5,5), strides=(2,2))(edge_conv_1)
edge_conv_2 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(edge_conv_2)

edge_conv_3 = Conv2D(filters=96, kernel_size=(5,5), strides=(2,2))(edge_conv_2)
edge_conv_3 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(edge_conv_3)
edge_conv_3 = BatchNormalization()(edge_conv_3)
edge_conv_3 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(edge_conv_3)

#Gray & Edge
gray_edge = Concatenate()([gray_conv_1, edge_conv_1])
gray_edge_conv_1 = Conv2D(filters=96, kernel_size=(5,5), strides=(2,2))(gray_edge)
gray_edge_conv_1 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(gray_edge_conv_1)

gray_edge_conv_2 = Conv2D(filters=96, kernel_size=(5,5), strides=(2,2))(gray_edge_conv_1)
gray_edge_conv_2 = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=None)(gray_edge_conv_2)
gray_edge_conv_2 = BatchNormalization()(gray_edge_conv_2)
gray_edge_conv_2 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(gray_edge_conv_2)

######################################

merge = Concatenate()([conv_3, gray_conv_3, edge_conv_3, gray_edge_conv_2])

# FC
flat = Flatten()(merge)

dense_1 = Dense(4096)(flat)
dense_1 = LeakyReLU()(dense_1)
dense_1 = Dropout(0.7)(dense_1) # 70%

dense_2 = Dense(4096)(dense_1)
dense_2 = LeakyReLU()(dense_2)
dense_2 = Dropout(0.7)(dense_2)

dense_3 = Dense(4096)(dense_2)
dense_3 = LeakyReLU()(dense_3)
dense_3 = Dropout(0.7)(dense_3)

output = Dense(2)(dense_3)

model = tf.keras.Model(inputs=input, outputs=output)

sgd = tf.keras.optimizers.experimental.SGD(learning_rate=0.001, weight_decay=0.0006, momentum=0.9, nesterov=True)
# adam = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=sgd,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


model.save("Models/art-medium_230x230_RGB-GRAY-EDGE.keras")
