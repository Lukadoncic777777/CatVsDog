import sys
import cv2
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

labels=[]
NUM_CATEGORIES=2
name=["Cat","Dog"]
IMG_WIDTH=200
IMG_HEIGHT=200
TEST_SIZE=0.6
EPOCHS=10
TRANSFORMER=transforms.Compose([transforms.Resize(IMG_WIDTH),transforms.CenterCrop((IMG_HEIGHT,IMG_WIDTH)),transforms.ToTensor()])
BATCH_SIZE=32
LR=0.001

device=torch.device("cpu")


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        self.dropout = nn.Dropout(0.5)
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y
        


def get_model(epochs,images,labels):
    datafile = MyDataset('train',images,labels,TRANSFORMER)                             
    dataloader = DataLoader(datafile, batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
    model = Model()
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()       
    # model=Model()
    cnt=0
    for epoch in range(epochs):
        running_loss = 0.0
        for img,label in dataloader:
            if cnt%20==0:
                print(cnt//20)
            cnt+=1
            img,label=Variable(img).to(device),Variable(label).to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs,label.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

    torch.save(model.state_dict(), 'model_new.pth')
    return model


class MyDataset(data.Dataset):

    def __init__(self,mode,images,labels,transformer):
        super().__init__()
        self.image=images
        self.mode=mode
        self.label=labels
        self.data_size=len(labels)
        self.transform=transformer
        #load data

    def __getitem__(self,index):
        if self.mode=='train':
            img = Image.open(self.image[index]).convert("RGB")
            label = self.label[index]
            return self.transform(img), torch.LongTensor([label])
        else:
            img = Image.open(self.image[index]).convert("RGB")
            return self.transform(img)

    def __len__(self):
        return self.data_size

def load_data(data_dir):

    images=[]
    labels=[]
    for label in range(0,NUM_CATEGORIES):
        nowdir=data_dir+os.sep+f'{name[label]}'
        for filename in os.listdir(nowdir):
            filepath=os.path.join(nowdir,filename)
            # img=cv2.imread(filepath)
            # img=cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
            images.append(filepath)
            labels.append(label)

    return (images,labels)

def main(args):

    model=None
    images,labels=load_data(sys.argv[2])
    X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=TEST_SIZE,random_state=2039)
    if sys.argv[1]=='0':    
        model=get_model(EPOCHS,X_train,y_train)
    else:
        model=Model()
        model.to(device)
        model.load_state_dict(torch.load(sys.argv[3]))

    model.eval()

    accepted=0
    wrong_answer=0

    test_size=len(X_test)


    for i in range(test_size):
        image=Image.open(X_test[i]).convert('RGB')
        input_tensor=TRANSFORMER(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output,dim=1).squeeze(0)
        ans=0
        if probabilities[0].item()<probabilities[1].item():
            ans=1
        if ans==y_test[i]:
            accepted+=1
        else:
            wrong_answer+=1
        print(f'all:{accepted+wrong_answer},AC:{accepted},WA:{wrong_answer}')

    return





if __name__=="__main__":
    main(sys.argv[1:])