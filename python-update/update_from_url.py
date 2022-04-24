import os
import numpy as np
from PIL import Image
from urllib.request import urlopen
from retinaface import RetinaFace
import cv2
from deepface import DeepFace
from distance import findCosineDistance
from update import update_user_faces, update_user_ff

def update(ROOT_dir, user_id, image_url):
  image_name = image_url.split("/")[-1]
  image_name = user_id + "_" + image_name.split(".")[0]

  backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
  models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "Ensemble" , "Facenet512"]

  # Load User with image in dictionary
  embedding_dir = os.path.join(ROOT_dir, "pipeline-deepface", "embedding")
  list_embedding_users = os.listdir(embedding_dir)
  dict_embedding_users = {}
  for embedding in list_embedding_users:
    emb = np.load(os.path.join(embedding_dir, embedding))
    id = embedding.split("_")[0]
    if id not in dict_embedding_users:
      dict_embedding_users[id] = [emb]
    else:
      dict_embedding_users[id].append(emb)
  user_embedding = dict_embedding_users[user_id]
  del dict_embedding_users[user_id]
  #Load image_url
  user_img = Image.open(urlopen(image_url))
  user_img= np.array(user_img)
  faces = RetinaFace.detect_faces(img_path = user_img, threshold=0.99)
  if type(faces) == dict:
    for (i,face) in enumerate(faces):
      x1,y1,x2,y2 = faces[face]["facial_area"]
      # y_incr = (y2 - y1)//4
      # x_incr = (x2 - x1)//5
      # y1 = max(y1-int(1.5*y_incr), 0)
      # x1 = max(x1-x_incr, 0)
      # y2 = y2+y_incr
      # x2 = x2+x_incr
      im_crop = user_img[y1:y2, x1:x2]
      im_crop = cv2.resize(im_crop, (224,224))
      
      face_embedding = DeepFace.represent(im_crop, model_name = models[8], enforce_detection=False)

      update_user_ff(ROOT_dir, user_id,im_crop[:,:,::-1], image_name, faces[face], i)
      vote = 0
      for embedding in user_embedding:
        if findCosineDistance(embedding, face_embedding) < 0.22:
          vote+=1
      if vote/len(user_embedding) > 0.5:
        update_user_faces(ROOT_dir, user_id, "None", im_crop[:,:,::-1], face_embedding, image_name, i)
        continue
      for user_related in dict_embedding_users:
        vote = 0
        for embedding in dict_embedding_users[user_related]:
          if findCosineDistance(embedding, face_embedding) < 0.22:
            vote+=1
        if vote/len(user_embedding) > 0.5:
          update_user_faces(ROOT_dir, user_related, user_id, im_crop[:,:,::-1], face_embedding, image_name, i)
          break