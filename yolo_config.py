import yaml
import argparse


def create_yaml(data_dic=None, path2yaml=''):
    '''


    :param data_dic:  data direcotry and class name to create yaml config
    :param path2yaml: path to save and same path use for train Yolov5 Model
    :return:

    '''
    data_yaml = data_dic

    with open(path2yaml, 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)

    with open("yolov5/data/data.yaml", 'r') as f:
        for line in f:
            print(line)


if __name__ == '__main__':
    data = dict(train='D:/SW_and_MANUALS/USER/PY_Projects/pcb-defect-detect-yolov5/tmp/images/train',
                # val='/kaggle/working/tmp/images/val',
                val='D:/SW_and_MANUALS/USER/PY_Projects/pcb-defect-detect-yolov5/tmp/images/test',
                nc=7,
                names=["background", "open", "short", "mousebite", "spur", "copper", "pin-hole"])
    create_yaml(data_dic=data, path2yaml="data/pcb_data.yaml")
