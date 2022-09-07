from deep_face.basemodels import Facenet

from deep_face.commons.functions import weight_file_path

def loadModel():
    model = Facenet.InceptionResNetV2(dimension = 512)
    weight_file = weight_file_path('facenet512_weights.h5')
    model.load_weights(weight_file)
    return model
