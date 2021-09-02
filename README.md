
# PCB Defect Detect 
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

https://user-images.githubusercontent.com/63192337/131648772-a859c404-e512-4534-b59f-d15a9fbcc478.mp4


This Repository present the Create Data set for train Model , how to train by using YoloV5 and test data by compare actual label and predict label.
Create main.py to run the whole model and detect the defect from PCB Image.

Image Size required = 640 x 640


:memo:  yoloV5 : 
---------------------------------------------------------------------------------
https://github.com/ultralytics/yolov5


:file_folder: ~~ Dataset :~~
---------------------------------------------------------------------------------


https://github.com/tangsanli5201/DeepPCB


:arrow_heading_down:  Download Model and Weight
---------------------------------------------------------------------------------
  - Download weight from weight Directory follow the link below
  - 
  - Download or clone YoloV5 model from above link.




:desktop_computer: 	:hammer_and_wrench: Download Required Library:
---------------------------------------------------------------------------------

:arrow_right:  matplotlib>=3.2.2

:arrow_right:  numpy>=1.18.5

:arrow_right:  opencv-python>=4.1.2

:arrow_right:  Pillow>=8.0.0

:arrow_right:  PyYAML>=5.3.1

:arrow_right:  scipy>=1.4.1

:arrow_right:  torch>=1.7.0

:arrow_right:  torchvision>=0.8.1

:arrow_right:  tqdm>=4.41.0


# :rocket:  Weights & Biases 


## bonous of  W&B use for monitor live performance of your model.

by using command :

```
!pip install -q --upgrade wandb
### # Login 
import wandb
wandb.login()
```

use above code in Notebook ---> Jupyter,Kaggle or Google colab 
