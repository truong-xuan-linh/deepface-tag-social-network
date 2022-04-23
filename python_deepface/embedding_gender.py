from deepface import DeepFace
import cv2
import os

from tqdm import trange
import numpy as np
def get_embedding_gender_image(img_dir, model):
    embedding_list = []
    gender_list = []
    image_list = []
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "Ensemble" , "Facenet512"]
    list_face_img = os.listdir(img_dir)
    for i in trange(len(list_face_img)):
      img_name = list_face_img[i]
      img = cv2.imread(os.path.join
      (img_dir, img_name))
      img = img[:,:,::-1]
      img_quality = np.expand_dims(cv2.resize(img, (224,224))/255.0, axis = 0)

      rs = model.predict(img_quality)
      if np.argmax(rs[0]) == 2:
        embedding_img = DeepFace.represent(img, model_name = models[8], enforce_detection =False)
        gender = DeepFace.analyze(img, actions = ['gender'], enforce_detection=False, detector_backend=backends[1])

        embedding_list.append(embedding_img)
        gender_list.append(gender)

        image_list.append(img_name)
    return image_list, embedding_list, gender_list