from deep_face.basemodels import VGGFace
from deep_face.commons.functions import weight_file_path
import numpy as np

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	import keras
	from keras.models import Model, Sequential
	from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
	from tensorflow import keras
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Convolution2D, Flatten, Activation


def loadModel():
	model = VGGFace.baseModel()

	#--------------------------

	classes = 101
	base_model_output = Sequential()
	base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	#--------------------------

	age_model = Model(inputs=model.input, outputs=base_model_output)

	#--------------------------

	#load weights
	weight_file = weight_file_path('age_model_weights.h5')
	age_model.load_weights(weight_file)

	return age_model

	#--------------------------

def findApparentAge(age_predictions):
	output_indexes = np.array([i for i in range(0, 101)])
	apparent_age = np.sum(age_predictions * output_indexes)
	return apparent_age
