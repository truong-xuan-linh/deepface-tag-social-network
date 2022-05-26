from retinaface import RetinaFace
import os
from tqdm import trange
import cv2
import pandas as pd

def face_extraction(user_id, image_dir, faces_db):
    # data = {"image_name" : [],
    #       "score" : [],
    #       "facial_area" : [],
    #         "right_eye" : [],
    #         "left_eye" : [],
    #         "nose_eye" : [],
    #         "mouth_right" : [],
    #         "mouth_left" : [],
    #       "crop_image_name" : []
    #       }
    data = []
    temp = image_dir.split("/")
    user = temp[-1]
    pos_dir = "/".join(temp[:-2]) + "/pipeline-deepface/full_face/" + user
    try :
      os.makedirs(pos_dir)
    except:
      pass

    # information_dir = "/".join(temp[:-2]) + "/pipeline-deepface/detail"
    # try :
    #   os.makedirs(information_dir)
    # except:
    #   pass

    full_face_dir = "/".join(temp[:-2]) + "/pipeline-deepface/full_face_image"
    try :
      os.makedirs(full_face_dir)
    except:
      pass

    list_image = os.listdir(image_dir)
    for im in trange(len(list_image)):
      img_name = list_image[im]
      try:
        faces = RetinaFace.detect_faces(img_path = os.path.join(image_dir,img_name), threshold=0.99)
      except:
        continue
      if type(faces) == dict: 
        image = cv2.imread(os.path.join(image_dir,img_name))
        w,h,_ = image.shape
        for (i,face) in enumerate(faces):

          x1,y1,x2,y2 = faces[face]["facial_area"]
          
          im_crop = image[y1:y2, x1:x2]
          #cv2_imshow(im_crop)
          if im_crop.shape[0]*im_crop.shape[1] > 80*80:
            name =img_name.split(".")[0] + "*"+str(i) + ".jpg"

            # data["image_name"].append(img_name.split(".")[0])
            # data["score"].append(faces[face]["score"])
            # data["facial_area"].append(faces[face]["facial_area"])
            # data["right_eye"].append(faces[face]["landmarks"]["right_eye"])
            # data["left_eye"].append(faces[face]["landmarks"]["left_eye"])
            # data["nose_eye"].append(faces[face]["landmarks"]["nose"])
            # data["mouth_right"].append(faces[face]["landmarks"]["mouth_right"])
            # data["mouth_left"].append(faces[face]["landmarks"]["mouth_left"])
            # data["crop_image_name"].append(name.split(".")[0]
            data.append({"image_name" : img_name.split(".")[0],
                        "crop_image_name": name.split(".")[0],
                        "upload_user_id": user_id,
                        "face_infor":{
                                      "score" : float(faces[face]["score"]),
                                      "facial_area" : list(map(float,faces[face]["facial_area"])),
                                        "right_eye" : list(map(float,faces[face]["landmarks"]["right_eye"])),
                                        "left_eye" : list(map(float,faces[face]["landmarks"]["left_eye"])),
                                        "nose" : list(map(float,faces[face]["landmarks"]["nose"])),
                                        "mouth_right" : list(map(float,faces[face]["landmarks"]["mouth_right"])),
                                        "mouth_left" : list(map(float, faces[face]["landmarks"]["mouth_left"]))
                        }
                        })
            cv2.imwrite(os.path.join(full_face_dir,name), cv2.resize(im_crop,(224,224)))
            cv2.imwrite(os.path.join(pos_dir,name), cv2.resize(im_crop,(224,224)))
          # if im_crop.shape[0]*im_crop.shape[1] > 80*80:
          #   name =img_name.split(".")[0] + "*"+str(i) + ".jpg"

          #   data["image_name"].append(img_name.split(".")[0])
          #   data["score"].append(faces[face]["score"])
          #   data["facial_area"].append(faces[face]["facial_area"])
          #   data["right_eye"].append(faces[face]["landmarks"]["right_eye"])
          #   data["left_eye"].append(faces[face]["landmarks"]["left_eye"])
          #   data["nose_eye"].append(faces[face]["landmarks"]["nose"])
          #   data["mouth_right"].append(faces[face]["landmarks"]["mouth_right"])
          #   data["mouth_left"].append(faces[face]["landmarks"]["mouth_left"])
          #   data["crop_image_name"].append(name.split(".")[0])
          #   cv2.imwrite(os.path.join(full_face_dir,name), cv2.resize(im_crop,(224,224)))
          #   cv2.imwrite(os.path.join(pos_dir,name), cv2.resize(im_crop,(224,224)))
    faces_db.insert_many(data)
    # df = pd.DataFrame(data)
    # df.to_csv(os.path.join(information_dir, user + "_ff.csv"), index=False)