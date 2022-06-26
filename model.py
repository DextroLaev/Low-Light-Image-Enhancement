import tensorflow as tf

def model():
	Input = tf.keras.Input(shape=(300,300,3))
	conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')(Input)
	conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')(conv1)
	conv3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')(conv2)
	conv4 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')(conv3)
	concat_3_4 = tf.keras.layers.Concatenate(axis=-1)([conv3,conv4])
	conv5 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')(concat_3_4)
	concat_2_5 = tf.keras.layers.Concatenate(axis=-1)([conv2,conv5])
	conv6 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same',activation='relu')(concat_2_5)
	concat_1_6 = tf.keras.layers.Concatenate(axis=-1)([conv1,conv6])
	last_layer = tf.keras.layers.Conv2D(24, kernel_size=(3, 3), strides=(1,1), padding='same',activation='tanh')(concat_1_6)
	m = tf.keras.Model(inputs=Input,outputs=last_layer)
	return m