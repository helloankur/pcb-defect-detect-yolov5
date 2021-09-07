import numpy
import os
import pandas
import yaml
import cv2
import csv
import glob
import os
import shutil
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import random

classes = {0: "background (not used)", 1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}



# folder_img_dir = glob.glob(path_ds + '\\*\\*')

class create_dataset:

    def __init__(self,splil=0.1,path_ds=" ",test_txt='',train_val_txt=""):   # split default 0.1 for training set

        self.split=splil
        self.test_txt=test_txt
        self.train_val_txt=train_val_txt
        self.path_ds=path_ds

        # path_ds = 'pcb_defect/PCBData/'


        folder_img_dir = glob.glob('pcb_defect/PCBData/**/*.jpg', recursive=True)
        self.df_all = []
        train_val = []
        self.test = []

        os.makedirs('tmp\\images\\train', exist_ok=True)
        os.makedirs('tmp\\images\\val', exist_ok=True)
        os.makedirs('tmp\\images\\test', exist_ok=True)
        os.makedirs('tmp\\labels\\train', exist_ok=True)
        os.makedirs('tmp\\labels\\val', exist_ok=True)
        os.makedirs('tmp\\labels\\test', exist_ok=True)

        with open(test_txt) as f:
            for line in f.readlines():
                self.test.append(line)

        with open(train_val_txt) as f1:
            for line in f1.readlines():
                train_val.append(line)

        val_split = int(len(train_val) * splil)

        random.seed(101)
        random.shuffle(train_val)

        self.train = train_val[val_split:]  # create train dataset from train_val text file
        self.val = train_val[:val_split]  # create val dataset from train_val text file

        # print(train)
        # print(val)

        self.datasets = [self.train, self.val, self.test]
        self.dir_nme = ['train', 'val', 'test']


    def to_csv(self,data):
        columns = list(data.keys())
        values = list(data.values())
        arr_len = len(values)
        df = pd.DataFrame(np.array(values, dtype=object).reshape(1, arr_len), columns=columns).reset_index()
        # print(df)
        return df



    def dataset4yolo(self):

        for k in tqdm(self.datasets):
            folder_num = (self.datasets.index(k))
            for i in tqdm(k):
                data = {}
                for j in i.split(" "):
                    print_buffer = []
                    if '.jpg' in j:
                        path_jpg = self.path_ds + j
                        head_img, tail_img = os.path.split(path_jpg)
                        # print(head)

                        org_name = (tail_img.strip('.jpg'))
                        # print(tail+'_test.jpg')

                        jpg_path = (head_img + '/' + org_name + '_test.jpg')
                        img_read = cv2.imread(jpg_path)
                        shutil.copy(jpg_path, 'tmp/images/' + str(self.dir_nme[folder_num]))
                        os.rename('tmp/images/' + str(self.dir_nme[folder_num]) + '/' + org_name + '_test.jpg',
                                  'tmp/images/' + str(self.dir_nme[folder_num]) + '/' + tail_img)

                    if '.txt' in j:
                        path_txt = self.path_ds + j.strip('\n')

                        with open(path_txt, 'r') as f:
                            for line in f.readlines():
                                annot = (line.strip("\n").split(" "))

                                x_min = int(annot[0])
                                y_min = int(annot[1])
                                x_max = int(annot[2])
                                y_max = int(annot[3])
                                label = int(annot[4])

                                # Transform the bbox co-ordinates as per the format required by YOLO v5
                                b_center_x = (x_min + x_max) / 2
                                b_center_y = (y_min + y_max) / 2
                                b_width = (x_max - x_min)
                                b_height = (y_max - y_min)

                                # Normalise the co-ordinates by the dimensions of the image
                                image_w, image_h, image_c = img_read.shape
                                b_center_x /= image_w
                                b_center_y /= image_h
                                b_width /= image_w
                                b_height /= image_h

                                print_buffer.append(
                                    "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(label, b_center_x, b_center_y, b_width, b_height))

                                head_txt, tail_txt = os.path.split(path_txt)
                                # print(tail_txt)

                                print("\n".join(print_buffer),
                                      file=open('tmp\\labels\\' + self.dir_nme[folder_num] + '\\' + tail_txt.strip('\n'), "w"))

                                id = tail_txt.strip('.txt')

                                data['id'] = id
                                data['label'] = label
                                data['x_min'] = x_min
                                data['y_min'] = y_min
                                data['x_max'] = x_max
                                data['y_max'] = y_max
                                data['b_center_x'] = b_center_x
                                data['b_center_y'] = b_center_y
                                data['b_width'] = b_width
                                data['b_height'] = b_height
                                data['image_w'] = image_w
                                data['image_h'] = image_h
                                data['split'] = self.dir_nme[folder_num]

                                # Convert Dic data to  dataframe

                                df = self.to_csv(data)
                                self.df_all.append(df)



        final_df = pd.concat(self.df_all, ignore_index=True)

        del final_df['index']

        #print(final_df)

        final_df.to_csv('meta.csv', index=False)

        print("Data set created Train data:" ,len(self.train))
        print("Data set created Validation data:", len(self.val))
        print("Data set created Test data:", len(self.test))
        print("Meta file create as :meta.csv")



if __name__ == '__main__':
    path_ds = 'DeepPCB/PCBData/'
    test_txt_path='DeepPCB\\PCBData\\test.txt'
    train_val_txt_path='DeepPCB\\PCBData\\trainval.txt'
    run=create_dataset(path_ds=path_ds,test_txt=test_txt_path,train_val_txt=train_val_txt_path)
    run.dataset4yolo()

