import os
import shutil

import cv2

from detect import pred
import matplotlib.pyplot  as plt
from cv_binary import img_process


def detection_start(test_img_read,tmp_img_read):



    test_img=cv2.imread(test_img_read, 0)
    temp_img=cv2.imread(tmp_img_read,0)

    w_test,h_test=test_img.shape

    if w_test !=512 and h_test !=512:
        test_img=cv2.resize(test_img,dsize=(512,512))

    cv2.imwrite('runs/pred_img.jpg', test_img)


    w_temp,h_temp=temp_img.shape

    if w_temp !=512 and h_temp !=512:
        test_img=cv2.resize(temp_img,dsize=(512,512))

    cv2.imwrite('runs/temp_img.jpg',test_img)

    y_pred_img,result=pred('runs/pred_img.jpg')

    fig = plt.figure(1, figsize=(15, 15))

    # Plot 2 image (PCB Defect  image)
    fig.add_subplot(121)
    plt.title('PCB Defect  image')
    pred_img = plt.imread(y_pred_img)
    plt.imshow(pred_img)


    # Plot 2 image (PCB Template  image)
    fig.add_subplot(122)
    plt.title('PCB Template  image')
    temp_img = plt.imread('runs/temp_img.jpg')
    plt.imshow(temp_img)
    plt.show(block=False)
    plt.pause(1)
    plt.show()

    try:
        shutil.rmtree('runs/detect', ignore_errors=True)
        for i in os.listdir('runs'):
            os.remove('runs/'+i)
    except:
        pass



if __name__ == '__main__':
    detection_start('DeepPCB/PCBData/group00041/00041/00041000_test.jpg',
                    'DeepPCB/PCBData/group00041/00041/00041000_temp.jpg')


