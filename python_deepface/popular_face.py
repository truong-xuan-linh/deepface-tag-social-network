from distance import findCosineDistance
from embedding_gender import get_embedding_gender_image


def most_popular_face(image_dir, quality_model, glass_model, threshold = 3):
    #Khởi tạo
    temp = image_dir.split("/")
    user = temp[-1]
    full_face_dir = "/".join(temp[:-1]) + "/" + user

    image_list, embedding_list, gender_list, glass_list = get_embedding_gender_image(full_face_dir, quality_model, glass_model)
    # image_list = image_list1.copy()
    # embedding_list = embedding_list1.copy()
    # gender_list = gender_list1.copy()
    list_person = []
    list_embedding_person = []
    list_glasses_person = []
    len_lst = []
    #Load model cho việc predict chất lượng ảnh

    while image_list != []:
      pos =  []
      name = []
      embedding_person = []
      glasses_person = []
      count = 1
      embedding_check = []
      #Load ảnh lấy ra so sánh

      name.append(image_list[0])
      embedding_person.append(embedding_list[0])
      glasses_person.append(glass_list[0])
      embedding_check.append(embedding_list[0])
      gender1 = gender_list[0]
      glass1 = glass_list[0] == 0
      del glass_list[0]
      del image_list[0]
      del embedding_list[0]
      del gender_list[0]

      
      #Nếu chất lượng ảnh 1 tốt thì lấy nó đi so sánh
      
      for (i,img_name) in enumerate(image_list):
        gender2 = gender_list[i]
        glass2 = glass_list[i] == 0
        img_embedding = embedding_list[i]
        thre = 0.28
        if glass1 == True or glass2 == True:
          thre =  0.18
        if gender1 == gender2:
          vote = 0
          need_replace = -1
          
          for (idx, em_check) in enumerate(embedding_check):
                #cv2_imshow(img_check)
            if findCosineDistance(em_check, img_embedding) <= thre:
              vote+=1
            else:
              need_replace = idx

          if len(embedding_check) < threshold:
            len_check = len(embedding_check) +1
          else:
            len_check = len(embedding_check)

          if vote >= len_check-1:
                #cv2_imshow(img2)
            embedding_person.append(img_embedding)
            glasses_person.append(glass_list[i])
            if count <threshold:
              pos.append(i)
              name.append(img_name)
              embedding_check.append(img_embedding)
              count+=1
            else:
              pos.append(i)
              name.append(img_name)
              if need_replace != -1:
                embedding_check[need_replace] = img_embedding
      list_person.append(name)
      list_embedding_person.append(embedding_person)
      list_glasses_person.append(glasses_person)
      for (j,p) in enumerate(pos):
        del image_list[p-j]
        del embedding_list[p-j]
        del gender_list[p-j]
        del glass_list[p-j]
    #############Vot du lieu###############
    for (_i, ps) in enumerate(list_embedding_person):
      list_embedding_check =ps.copy() # random.sample(ps, min(len(ps), 3))
      glasses = list_glasses_person[_i].copy()
      for (_j, ps2) in enumerate(list_embedding_person[_i+1:]):
        for (k, p) in enumerate(ps2): 
          vote = 0
          glass1 = list_glasses_person[_i+1+_j][k].copy()
          for (h, embedding_check) in enumerate(list_embedding_check):
            thre = 0.28
            if glass1 == 0 or glasses[h] == 0:
              thre = 0.18
            if findCosineDistance(embedding_check, p) <= thre:
              vote+=1

          # if vote/len(list_embedding_check) > 0.6:#len(list_embedding_check) < 10:
          #   len_check = len(list_embedding_check) +1
          # else:
          #   len_check = len(list_embedding_check)

          if vote/(len(list_embedding_check)+1) > 0.7:

            list_person[_i]  = list_person[_i] + list_person[_j+_i+1]
            list_embedding_person[_i]  = list_embedding_person[_i] + list_embedding_person[_j+_i+1]
            list_glasses_person[_i]  = list_glasses_person[_i] + list_glasses_person[_j+_i+1]

            list_embedding_person[_j+_i+1] = []
            list_glasses_person[_j+_i+1] = []
            list_person[_j+_i+1] = []
            break
    #############vot du lieu###########
    for (i,ps) in enumerate(list_person):
      len_lst.append(len(ps))
    return list_person, len_lst, list_embedding_person