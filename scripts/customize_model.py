import torchvision.models as models
from torchsummary import summary
import torch
#import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

class customize_model(nn.Module):
  def __init__(self, input_size=128, sel_model='resnet18'):
    super(customize_model, self).__init__()

    ## Select model
    if sel_model == 'resnext50':
        self.model_ = models.resnext50_32x4d(pretrained=True)
        self.model_.conv1.weight = nn.Parameter(self.model_.conv1.weight.sum(dim=1).unsqueeze(1))
    elif sel_model == 'vgg16':
        model_0 = models.vgg16_bn(pretrained=True) 
        self.model_ = nn.Sequential(*list(model_0.children())[0]) #VGG16_bn 
        self.model_[0].weight = nn.Parameter(self.model_[0].weight.sum(dim=1).unsqueeze(1))
    elif sel_model == 'wide50':
        self.model_ = models.wide_resnet50_2(pretrained=True)
        self.model_.conv1.weight = nn.Parameter(self.model_.conv1.weight.sum(dim=1).unsqueeze(1))
    elif sel_model == 'mobilev2':
        self.model_ = models.mobilenet_v2(pretrained=True)
        self.model_.features[0][0].weight = nn.Parameter(self.model_.features[0][0].weight.sum(dim=1).unsqueeze(1))
    elif sel_model == 'densenet121':
        self.model_ = models.densenet121(pretrained=True)
        self.model_.features.conv0.weight = nn.Parameter(self.model_.features.conv0.weight.sum(dim=1).unsqueeze(1))
    else:
        self.model_ = models.resnet18(pretrained=True)
        self.model_.conv1.weight = nn.Parameter(self.model_.conv1.weight.sum(dim=1).unsqueeze(1))        

    for i, param in enumerate(self.model_.parameters()):
        param.requires_grad = True #False
        print(i, param.requires_grad)
    if sel_model =='vgg16':
        self.f_resnet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=np.int(512*4*4), out_features=2048, bias=True), #wide_resnet50-2 512*256
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=np.int(512*2*2), out_features=1000, bias=True), #wide_resnet50-2 512*256
            nn.Linear(in_features=np.int(1000), out_features=10, bias=True)
            )    
    else:
        self.f_resnet = nn.Sequential(
            nn.Linear(in_features=np.int(1000), out_features=10, bias=True)
  )   

  def forward(self, input):

    midlevel_features = self.model_(input)
    output = self.f_resnet(midlevel_features)
    return output

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #for gpu
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    #pl.seed_everything(0)

    model = models.vgg16_bn(pretrained=False, num_classes=10)
    model = model.to(device) #for gpu
    print('Gray_model',model)

    dim = (3,128,128)
    summary(model,dim)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time)) 
