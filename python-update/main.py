import os
from update_from_url import update
import argparse
import pymongo

def main(user_id, image_url, uri, user_face_db, faces_db):
  ROOT_dir = os.environ["DEEPFACE_DATA"]
  update(ROOT_dir, user_id, image_url, user_face_db, faces_db)

if __name__ == '__main__':
  uri = "mongodb+srv://truong-xuan-linh:hahalolo@deepface.ky81b.mongodb.net/test"
  client = pymongo.MongoClient(uri)
  mydb = client["test"]
  user_face = mydb["user-face"]
  faces = mydb["faces"]

  parser = argparse.ArgumentParser()
  parser.add_argument("--user_id", type=str, help="user id")
  parser.add_argument("--url", type=str, help="url image")
  args = parser.parse_args()

  main(args.user_id, args.url, uri, user_face, faces)