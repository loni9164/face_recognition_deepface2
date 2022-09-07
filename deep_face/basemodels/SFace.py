import numpy as np
import cv2 as cv
from deep_face.commons.functions import weight_file_path


class _Layer:
    input_shape = (None, 112, 112, 3)
    output_shape = (None, 1, 128)


class SFaceModel:

    def __init__(self, model_path):
        self.model = cv.FaceRecognizerSF.create(
            model=model_path,
            config="",
            backend_id=0,
            target_id=0)

        self.layers = [_Layer()]

    def predict(self, image):
        # Preprocess
        input_blob = (image[0] * 255).astype(
            np.uint8)  # revert the image to original format and preprocess using the model

        # Forward
        embeddings = self.model.feature(input_blob)

        return embeddings


def load_model():
    weight_file = weight_file_path('face_recognition_sface_2021dec.onnx')
    print(weight_file)
    model = SFaceModel(model_path = weight_file)
    return model

