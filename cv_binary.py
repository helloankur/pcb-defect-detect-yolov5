import cv2



def img_process(img_path):
    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh



if __name__ == '__main__':
    img=img_process('pcb1.jpg')
    cv2.imshow('1',img)
    cv2.waitKey()
