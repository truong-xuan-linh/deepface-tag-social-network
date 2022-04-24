import os
from update_from_url import update
import argparse


def main(user_id, image_url):
  ROOT_dir = os.environ["DEEPFACE_DATA"]
  update(ROOT_dir, user_id, image_url)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--user_id", type=str, help="user id")
  parser.add_argument("--url", type=str, help="url image")
  args = parser.parse_args()

  main(args.user_id, args.url)