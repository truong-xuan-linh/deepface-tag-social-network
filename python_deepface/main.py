# -*- coding: utf-8 -*-
"""Backup1 of Backup of Backup_Find Face.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wUrrGkerinCrTJyel8Sga4rbtWsb0tgE

#**TASK 1:**

##**Import**
"""


#import face_recognition
import cv2
import gdown
import os
from tqdm import tqdm, trange
from keras.models import load_model
from distance import findCosineDistance
from embedding_gender import get_embedding_gender_image
from face_extraction import face_extraction
from popular_face import most_popular_face
from take_object import take_object
import pandas as pd
from subprocess import call
import pymongo
# import insightface

"""##**Face extraction**"""

def main(ROOT_dir, quality_model_dir, glass_model_dir, uri):
  
  client = pymongo.MongoClient(uri)
  mydb = client["test"]
  user_face = mydb["user-face"]
  faces = mydb["faces"]
  quality_model = load_model(quality_model_dir)
  glass_model = load_model(glass_model_dir)
  BIG_image_dir = os.path.join(ROOT_dir, "images")
  list_user_id = os.listdir(BIG_image_dir)
  list_user_id.sort()
  done = False
  while not done:
    if len(list_user_id) >= 2:
      for user_id in list_user_id[:-1]:
      #####
        print(f"{user_id} FACE EXTRACTION RUNNING............")
        face_extraction(user_id, os.path.join(BIG_image_dir, user_id), faces)
        BIG_full_face = os.path.join(ROOT_dir, "pipeline-deepface/full_face")
        #list_full_face = os.listdir(BIG_full_face)
        image_dir = os.path.join(BIG_full_face, user_id)
        print(f"{user_id} FIND MOST POPULAR FACE RUNNING............")
        list_person, len_lst, list_embedding_person = most_popular_face(image_dir, quality_model, glass_model)
        try:
          len_lst, list_person, list_embedding_person = zip(*sorted(zip(len_lst, list_person, list_embedding_person), reverse= True))
          top_len = len_lst[0]
          top_user = list_person[0][0].split("_")[0]
          print(f"Find {top_len} face(s) of user {top_user}")
          take_object(ROOT_dir, top_user, image_dir, list_person[0], list_embedding_person[0], user_face)
        except:
          print(f"ERROR: Can't find valid face in user {user_id}")
        try:
          call('rm ' + image_dir+ ' -r', shell=True)
          call('rm ' + BIG_image_dir + '/' + user_id+ ' -r', shell=True)
        except:
          print(f"ERROR: Can't delete {user_id} folder")
    list_user_id = os.listdir(BIG_image_dir)
    list_user_id.sort()
    if "zzzz" in list_user_id and len(list_user_id) == 1:
      done = True
      print("DONE!!!")
      call('rm ' + BIG_image_dir + '/' + "zzzz"+ ' -r', shell=True)

if __name__ == '__main__':
  ROOT_dir = os.getenv('DEEPFACE_DATA')
  try: 
    os.mkdir(os.path.join(ROOT_dir, "pipeline-deepface"))
  except:
    pass
  uri = "mongodb+srv://truong-xuan-linh:hahalolo@deepface.ky81b.mongodb.net/test"
  
  # a file
  quality_model_dir = os.path.join(ROOT_dir, "pipeline-deepface", "quality_model.h5")
  if "quality_model.h5" not in os.listdir(os.path.join(ROOT_dir, "pipeline-deepface")):
    url = "https://drive.google.com/uc?id=1-GztQP5wlqfjz1llOBZrLqV-DcWip24S"
    gdown.download(url, quality_model_dir, quiet=False)

  glass_model_dir = os.path.join(ROOT_dir, "pipeline-deepface", "glass_model.h5")
  if "glass_model.h5" not in os.listdir(os.path.join(ROOT_dir, "pipeline-deepface")):
    url = "https://drive.google.com/uc?id=1-Dm6zATOGiGtiqPVvL88N2--CeIqqxv1"
    gdown.download(url, glass_model_dir, quiet=False)
  main(ROOT_dir, quality_model_dir, glass_model_dir, uri)