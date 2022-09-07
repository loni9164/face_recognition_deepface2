
from deep_face import DeepFace
img1_path = r'C:\Users\PC\OneDrive\Documents\face_recognition\prakash_1.png'
img2_path = r'C:\Users\PC\OneDrive\Documents\face_recognition\prakash_2.png'


result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name = 'SFace',detector_backend = 'opencv', distance_metric='euclidean')
print(result)

#obj = DeepFace.analyze(img_path = img2_path, actions = ['age', 'gender'])
#print(obj)










