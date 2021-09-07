import glob
import os
import shutil
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import random

classes = {0: "background (not used)", 1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}





class create_dataset:

    def __init__(self,val_split=0.1,test_split=None,path_ds=" "):   # split default 0.1 for training set


        self.val_split=val_split
        self.test_split=test_split


        self.path_ds=path_ds

        self.df_all = []

        self.train=[]
        self.train_val = []
        self.test = []
        self.val=[]




    def to_csv(self,data):
        columns = list(data.keys())
        values = list(data.values())
        arr_len = len(values)
        df = pd.DataFrame(np.array(values, dtype=object).reshape(1, arr_len), columns=columns).reset_index()
        # print(df)
        return df

    def directory_split(self):

        label_list = ('dataset\\txt')
        img_list = ('dataset\\img')

        sample_label = (os.listdir(label_list))
        sample_img = (os.listdir(img_list))

        print(len(sample_label))
        print(len(sample_img))

        random.seed(101)
        random.shuffle(sample_img)

        split_size = int(len(sample_img) * self.test_split)
        self.train_val = sample_img[split_size:]

        self.test = sample_img[:split_size]

        val_split = int(len(self.train_val) * self.val_split)
        random.seed(101)
        random.shuffle(self.train_val)
        self.train = self.train_val[val_split:]  # create train dataset from train_val text file
        self.val = self.train_val[:val_split]

        # print(len(self.train))
        # print((len(self.test)))
        # print(len(self.val))

        os.makedirs('tmp\\images\\train', exist_ok=True)
        os.makedirs('tmp\\images\\val', exist_ok=True)
        os.makedirs('tmp\\images\\test', exist_ok=True)
        os.makedirs('tmp\\labels\\train', exist_ok=True)
        os.makedirs('tmp\\labels\\val', exist_ok=True)
        os.makedirs('tmp\\labels\\test', exist_ok=True)

        self.datasets = [self.train, self.val, self.test]
        self.dir_nme = ['train', 'val', 'test']


    def one_dir_data_set(self):

        os.makedirs("dataset\\img", exist_ok=True)
        os.makedirs("dataset\\txt", exist_ok=True)
        folder_img_dir = glob.glob(path_ds + '\\*\\*',recursive=True)

        for i in tqdm(folder_img_dir):
            try:
                img_path = os.listdir(i)
                for tr_img in img_path:
                    # print(tr_img)
                    if 'test' in tr_img:
                        # print(img_path)
                        # print(i)
                        com_path = i + "\\" + tr_img
                        shutil.copy(src=com_path, dst='dataset\\img')
                        # print(com_path)

                    elif '.txt' in tr_img:
                        label_path = i + "\\" + tr_img
                        shutil.copy(label_path, 'dataset\\txt')
            except NotADirectoryError:
                print("Unknown file type ,required JPG or TXT format")

            except PermissionError:
                print("Permission Error ")

            except:pass

        self.directory_split()

    def dataset4yolo(self):
        for k in tqdm(self.datasets):
            folder_num = (self.datasets.index(k))
            for i in tqdm(k):
                data = {}
                print_buffer = []

                path_jpg = 'dataset/img/' + i
                # print(path_jpg)
                head_img, tail_img = os.path.split(path_jpg)
                img_id = (tail_img.strip('_test.jpg'))
                #print(img_id)
                # print(tail+'_test.jpg')
                img_read = cv2.imread(path_jpg)
                shutil.copy(path_jpg, 'tmp/images/' + str(self.dir_nme[folder_num]))

                os.rename('tmp/images/' + str(self.dir_nme[folder_num]) + '/' + img_id + '_test.jpg',
                          'tmp/images/' + str(self.dir_nme[folder_num]) + '/' + img_id+ '.jpg')



                path_txt = 'dataset/txt/' + (i.replace('_test.jpg', '.txt'))

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

        # print(final_df)

        final_df.to_csv('meta.csv', index=False)

        print("Data set created Train data:" ,len(self.train))
        print("Data set created Validation data:", len(self.val))
        print("Data set created Test data:", len(self.test))
        print("Meta file create as :meta.csv")



if __name__ == '__main__':
    path_ds = 'DeepPCB/PCBData/'
    test_txt_path='DeepPCB\\PCBData\\test.txt'
    train_val_txt_path='DeepPCB\\PCBData\\trainval.txt'
    run=create_dataset(test_split=0.33, path_ds=path_ds)

    run.one_dir_data_set()

    run.dataset4yolo()
