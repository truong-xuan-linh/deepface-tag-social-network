import numpy as np
import cv2
import pandas as pd
import os
from deepface import DeepFace


def update_user_faces(ROOT_dir, user_id, user_related, image, embedding, image_name, i):
  user_dir = os.path.join(ROOT_dir, "pipeline-deepface", "users")
  embedding_dir = os.path.join(ROOT_dir, "pipeline-deepface", "embedding")
  np.save(os.path.join(embedding_dir, user_id + "_" + image_name +"*"+ str(i)+ ".npy"), embedding)
  cv2.imwrite(os.path.join(user_dir, user_id + "_" + image_name +"*"+ str(i)+ ".jpg"), image)
  face_dir = os.path.join(ROOT_dir,"pipeline-deepface", "detail", user_id + "_faces.csv")
  df = pd.read_csv(face_dir)
  data = dict()
  obj = DeepFace.analyze(image, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False, prog_bar= False)

  data["Image name"] = [image_name]
  data["Image crop name"] = [image_name + "*"+ str(i)]
  data["Age"] = [obj["age"]]
  data["Gender"] = [obj["gender"]]
  data["Race"] = [obj["race"]]
  data["Emotion"] = [obj["emotion"]]
  data["Relation"] = [user_related]
  df2 = pd.DataFrame(data)
  df = df.append(df2)
  df.to_csv(face_dir, index=False)
  print(f"UPDATED user, embbeding and faces {user_id}")

def update_user_ff(ROOT_dir, user_id, image, image_name, face, i):
  ff_image_dir = os.path.join(ROOT_dir,"pipeline-deepface", "full_face_image")
  cv2.imwrite(os.path.join(ff_image_dir, user_id + "_" + image_name +"*"+ str(i)+ ".jpg"), image)
  
  ff_dir = os.path.join(ROOT_dir, "pipeline-deepface", "detail", user_id + "_ff.csv")
  df = pd.read_csv(ff_dir)
  data = {}
  data["Image name"] = [image_name]
  data["Score"] = [face["score"]]
  data["Facial area"] = [face["facial_area"]]
  data["Right eye"] = [face["landmarks"]["right_eye"]]
  data["Left eye"] = [face["landmarks"]["left_eye"]]
  data["Nose eye"] = [face["landmarks"]["nose"]]
  data["Mouth right"] = [face["landmarks"]["mouth_right"]]
  data["Mouth left"] = [face["landmarks"]["mouth_left"]]
  data["Crop image name"] = [image_name + "*"+ str(i)]

  df2 = pd.DataFrame(data)
  df = df.append(df2)
  df.to_csv(ff_dir, index=False)
  print(f"UPDATED ff {user_id}")