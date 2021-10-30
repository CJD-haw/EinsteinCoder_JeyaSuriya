import time
import random
import datetime
import os
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import efficientnet_pytorch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(2021)

class DataLoaderConfig:
    batch_size = 32
    num_workers = 8

class TrainConfig:
    criterion = nn.CrossEntropyLoss 
    n_epochs = 10
    lr = 0.001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dir = 'data/train'
test_dir = 'data/test'

classes = sorted(os.listdir(train_dir))
print(classes)

N = [0,1]
normal_mapping = dict(zip(classes,N)) 
reverse_mapping = dict(zip(N,classes)) 

train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(40),              # resize shortest side
        transforms.CenterCrop(40),          # crop longest side
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = ImageFolder(train_dir, transform = train_transform)
testset = ImageFolder(test_dir, transform = train_transform)
# print(trainset)
# print(testset)

def show_image(img, label):
    print('Label: ', trainset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1,2,0))

show_image(*trainset[20])

torch.manual_seed(10)
val_size = len(trainset)//10
test_size = len(testset)
train_size = len(trainset) - val_size
train_ds, val_ds = random_split(trainset, [train_size, val_size])
test_ds = testset
# print(len(train_ds), len(val_ds), len(test_ds))

batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size, num_workers=4, pin_memory=True)

for images, labels in train_loader:
    fig, ax = plt.subplots(figsize=(18,10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    break

class DatasetRetriever(Dataset):
    def __init__(self, X, y, transforms=None):
        super().__init__()
        self.X = X.reshape(-1,40,40).astype(np.float32)
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index):
        image, target = self.X[index], self.y[index]
        image = np.stack([image] * 3, axis=-1)
        image /= 255.
        if self.transforms:
            image = self.transforms(image=image)['image']
            
        return image, torch.tensor(target, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]

def get_train_transforms():
    return A.Compose(
        [
            A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            A.Cutout(num_holes=8, max_h_size=2, max_w_size=2, fill_value=0, p=0.5),
            A.Cutout(num_holes=8, max_h_size=1, max_w_size=1, fill_value=1, p=0.5),
            A.Resize(40,40, p=1.),
            ToTensorV2(p=1.0),
        ], 
        p=1.0)

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(40,40, p=1.),
            ToTensorV2(p=1.0),
        ], 
        p=1.0
    )

plt.figure(figsize=(12,12))

for i in range(16):    
    image, target = train_ds[random.randint(0,len(train_ds))]
    numpy_image = image.permute(1,2,0).numpy()

    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(reverse_mapping[target],fontsize=12)
    plt.imshow(numpy_image);


class LossMeter:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AccMeter:
    def __init__(self):
        self.true_count = 0
        self.all_count = 0
        self.avg = 0
        
    def update(self, y_true, y_pred):
        y_true = y_true
        y_pred = y_pred.argmax(axis=1)
        self.true_count += (y_true == y_pred).sum()
        self.all_count += y_true.shape[0]
        self.avg = self.true_count / self.all_count

class Fitter:
    def __init__(
        self, model, device, criterion, n_epochs, 
        lr, sheduler=None, scheduler_params=None
    ):
        self.epoch = 0
        self.n_epochs = n_epochs
        self.base_dir = './'
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = np.inf
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if sheduler:
            self.scheduler = sheduler(self.optimizer, **scheduler_params)
            
        self.criterion = criterion().to(self.device)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, valid_loader):
        for e in range(self.n_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            self.log(f'\n{datetime.datetime.utcnow().isoformat()}\nLR: {current_lr}')

            t = int(time.time())
            summary_loss, final_scores = self.train_one_epoch(train_loader)
            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'final_score: {final_scores.avg:.5f}, ' + \
                f'time: {int(time.time()) - t} s'
            )

            t = int(time.time())
            summary_loss, final_scores = self.validation(valid_loader)
            self.log(
                f'[RESULT]: Valid. Epoch: {self.epoch}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'final_score: {final_scores.avg:.5f}, ' + \
                f'time: {int(time.time()) - t} s'
            )
            
            f_best = 0
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                f_best = 1

            self.scheduler.step(metrics=summary_loss.avg)    
            self.save(f'{self.base_dir}/last-checkpoint.bin')
            
            if f_best:
                self.save(f'{self.base_dir}/best-checkpoint.bin')
                print('New best checkpoint')

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = LossMeter()
        final_scores = AccMeter()
        
        t = int(time.time())
        for step, (images, targets) in enumerate(val_loader):
            print(
                f'Valid Step {step}/{len(val_loader)}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'final_score: {final_scores.avg:.5f}, ' + \
                f'time: {int(time.time()) - t} s', end='\r'
            )
            
            with torch.no_grad():
                targets = targets.to(self.device)
                images = images.to(self.device)
                batch_size = images.shape[0]
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss, final_scores

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = LossMeter()
        final_scores = AccMeter()
        
        t = int(time.time())
        for step, (images, targets) in enumerate(train_loader):
            print(
                f'Train Step {step}/{len(train_loader)}, ' + \
                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                f'final_score: {final_scores.avg:.5f}, ' + \
                f'time: {int(time.time()) - t} s', end='\r'
            )
            
            targets = targets.to(self.device)
            images = images.to(self.device)
            batch_size = images.shape[0]
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            final_scores.update(targets, outputs.detach())
            summary_loss.update(loss.detach().item(), batch_size)
            self.optimizer.step()

        return summary_loss, final_scores
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

def get_net():
    net = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b7')
    net._fc = nn.Linear(in_features=2560, out_features=2, bias=True)
    return net

net = get_net().to(DEVICE)

fitter = Fitter(
    model=net, 
    device=DEVICE, 
    criterion=TrainConfig.criterion, 
    n_epochs=TrainConfig.n_epochs, 
    lr=TrainConfig.lr, 
    sheduler=TrainConfig.scheduler, 
    scheduler_params=TrainConfig.scheduler_params
)

fitter.fit(train_loader, valid_loader)

checkpoint = torch.load('working/best-checkpoint.bin')
net.load_state_dict(checkpoint['model_state_dict']);
net.eval();

class DatasetRetriever(Dataset):
    def __init__(self, X, transforms=None):
        super().__init__()
        self.X = X.reshape(-1,40,40).astype(np.float32)
        self.transforms = transforms

    def __getitem__(self, index):
        image = self.X[index]
        image = np.stack([image] * 3, axis=-1)
        image /= 255.
        if self.transforms:
            image = self.transforms(image=image)['image']
            
        return image

    def __len__(self):
        return self.X.shape[0]

y=[]
result = []
for images, labels in test_loader:
    images2=images.to(DEVICE)
    y_pred = net(images2).detach().cpu().numpy().argmax(axis=1).astype(int) 
    result.extend(y_pred)
    y.extend(labels)
    
print(result[0:10])
print(y[0:10])

plt.figure(figsize=(12,12))

for i in range(16):    
    image, target = test_ds[i]
    numpy_image = image.permute(1,2,0).numpy()
    plt.subplot(4,4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{reverse_mapping[result[i]]}=={reverse_mapping[target]}', fontsize=12)
    plt.imshow(numpy_image)

ANS=y
PRED=result
accuracy=accuracy_score(ANS,PRED)
print(accuracy)


