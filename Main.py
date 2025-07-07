import sys
import cv2
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
import torch.utils.data as data

labels=[]
NUM_CATEGORIES=2
name={"Cat": 0, "Dog": 1}
IMG_WIDTH=300
IMG_HEIGHT=300
TEST_SIZE=0.4

class Model(nn.Module):
    
    def __init__(self):
        super(Model,self).__init__()

        

class MyDataset(data.Dataset):

    def __init__(self,mode,data_dir):
        super().__init__()
        self.image=[]
        self.mode=mode
        self.label=[]
        self.data_size=0

        #load data
        images=[]
        labels=[]
        for label in range(0,NUM_CATEGORIES):
            nowdir=data_dir+os.sep+f'{name[label]}'
            for filename in os.listdir(nowdir):
                filepath=os.path.join(nowdir,filename)
                img=cv2.imread(filepath)
                img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
                images.append(img)
                labels.append(label)
                self.data_size+=1
        self.image=images
        self.label=labels

    def __getitem__(self, index):
        return (self.image[index],self.label[index])

    def __len__(self):
        return self.data_size


def main(args):

    x_train,x_test,y_train,y_test=train_test_split()



    return





if __name__=="__main__":
    main(sys.argv[1:])