import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dense, Flatten, Reshape, Concatenate, LeakyReLU, Activation, Dropout
from keras.initializers import RandomNormal
from keras import Model, Input
import keras
import consts

def build_discriminator():
	init = RandomNormal(stddev=0.02) # Helps at training stabilization

	sketch_input = Input(shape=consts.input_shape)
	image_input = Input(shape=consts.output_shape)

	merged = Concatenate()([sketch_input, image_input])

	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)

	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	output = Activation('sigmoid')(d)

	model = Model([sketch_input, image_input], output)

	opt = keras.optimizers.adam_v2.Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

	return model

def encoder_block(inputs, filters, batchnorm=True):
	x = Conv2D(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(inputs)

	if batchnorm:
		x = BatchNormalization()(x, training=True)

	return LeakyReLU(alpha=0.2)(x)
 
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	x = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(layer_in)
	x = BatchNormalization()(x, training=True)

	if dropout:
		x = Dropout(0.5)(x, training=True)
	
	x = Concatenate()([x, skip_in])
	x = Activation('relu')(x)

	return x
 
def build_generator():
	initial_kernel = RandomNormal(stddev=0.02) # Helps at training stabilization

	sketch_input = Input(shape=consts.input_shape)
	
	e1 = encoder_block(sketch_input, 64, batchnorm=False)
	e2 = encoder_block(e1, 128)
	e3 = encoder_block(e2, 256)
	e4 = encoder_block(e3, 512)
	e5 = encoder_block(e4, 512)
	e6 = encoder_block(e5, 512)
	e7 = encoder_block(e6, 512)
	
	bottleneck = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_kernel)(e7)
	bottleneck = Activation('relu')(bottleneck)
	
	d1 = decoder_block(bottleneck, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=initial_kernel)(d7)
	output = Activation('tanh')(g)

	return Model(sketch_input, output)

def build_gan(g_model, d_model):
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	
	sketch_input = Input(shape=consts.input_shape)
	g_out = g_model(sketch_input)
	d_out = d_model([sketch_input, g_out])
	gan = Model(sketch_input, [d_out, g_out])

	opt = keras.optimizers.adam_v2.Adam(learning_rate=0.0002, beta_1=0.5)
	gan.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])

	return gan

