import shutil

import cv2

from detect import pred
import matplotlib.pyplot  as plt
from cv_binary import img_process


def detection_start(img_read):

    img=cv2.imread(img_read,0)

    w,h=img.shape

    if w !=640 and h !=640:
        img=cv2.resize(img,dsize=(640,640))

    cv2.imwrite('pred_img.jpg',img)

    y_pred_img,result=pred('pred_img.jpg')

    pred_img=plt.imread(y_pred_img)
    plt.imshow(pred_img)
    plt.show()
    shutil.rmtree('runs',ignore_errors=True)
    shutil.rmtree('pred_img.jpg',ignore_errors=True)


if __name__ == '__main__':
    detection_start('tmp/images/test/00041002.jpg')


