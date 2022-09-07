from deep_face.basemodels import VGGFace
from deep_face.commons.functions import weight_file_path


import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	from keras.models import Model, Sequential
	from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Convolution2D, Flatten, Activation


def loadModel():

	model = VGGFace.baseModel()

	#--------------------------

	classes = 2
	base_model_output = Sequential()
	base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	#--------------------------

	gender_model = Model(inputs=model.input, outputs=base_model_output)

	#--------------------------

	#load weights
	weight_file = weight_file_path('gender_model_weights.h5')
	gender_model.load_weights(weight_file)
	return gender_model

	#--------------------------
