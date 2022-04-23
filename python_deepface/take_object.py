import cv2
import os
from deepface import DeepFace
import pandas as pd
import numpy as np
def take_object(ROOT_dir, image_dir, list_tag_person, list_embedding_person):
    data = {"Image name": [],
          "Image crop name": [],
          "Age": [],
          "Gender": [],
          "Race": [],
          "Emotion": [],
          "Relation": []
          }
    
    temp = image_dir.split("/")
    user = temp[-1]
    user_dir = os.path.join(ROOT_dir, "pipeline-deepface/users")
    embedding_dir = os.path.join(ROOT_dir, "pipeline-deepface/embedding")
    try :
      os.makedirs(user_dir)
    except:
      pass
    try :
      os.makedirs(embedding_dir)
    except:
      pass
    
    for (j, i) in enumerate(list_tag_person):
      img = cv2.imread(os.path.join(image_dir, i))
      e_name = i.split(".")[0] + ".npy"
      img = cv2.resize(img, (224,224))
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      obj = DeepFace.analyze(img_rgb, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False, prog_bar= False)
      img_name, face_id = i.split("*")
      data["Image name"].append(img_name)
      data["Image crop name"].append(i.split(".")[0])
      data["Age"].append(obj["age"])
      data["Gender"].append(obj["gender"])
      data["Race"].append(obj["race"])
      data["Emotion"].append(obj["emotion"])
      data["Relation"].append("None")
      np.save(os.path.join(embedding_dir, e_name), list_embedding_person[j])
      cv2.imwrite(os.path.join(user_dir, i), img)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(os.path.join(ROOT_dir, "pipeline-deepface/detail"), user + "_faces.csv"))