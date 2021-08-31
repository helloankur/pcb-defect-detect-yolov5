import os
import time

import  cv2
import matplotlib.pyplot as plt

import matplotlib.image as mpimg


def test_actual_img(img_path,text_path):
    color={0: (255,255,0), 1: (255,125,0),
               2: (125,125,0), 3: (129,100,10), 4: (100,125,250), 5: (25,125,170), 6: (99,199,159)}

    classes = {0: "background (not used)", 1: "open",
               2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}

    img=cv2.imread(img_path)

    with open(text_path, 'r') as f:

      for line in f.readlines():

        annot = (line.strip("\n").split(" "))

        x_min = int(annot[0])
        y_min = int(annot[1])
        x_max = int(annot[2])
        y_max = int(annot[3])
        label = int(annot[4])

        img=cv2.rectangle(img,(x_min,y_min),(x_max,y_max),color=color[label],thickness=2)
        img_show=cv2.putText(img,classes[label],(int(x_min-10),int(y_min-10)),cv2.FONT_HERSHEY_TRIPLEX,1,
                             color=(255,0,0),thickness = 1)




    return  img_show


img_path='tmp\\images\\test\\'
txt_path='dataset\\txt\\'
#print(os.listdir(img_path))





for _ in os.listdir(img_path):
    text_file=(_.strip('.jpg'))+'.txt'
    img_read=img_path + _
    annot_path=txt_path+text_file
    print(img_read)
    print(annot_path)
    img_show=test_actual_img(img_read,annot_path)

    fig = plt.figure(1, figsize=(100, 100))
    fig.add_subplot(121)
    plt.imshow(img_show)

    # Plot 2 image (Predict image)
    fig.add_subplot(122)


    plt.show()




