import torchvision
import torch.nn as nn
from utils.func import *

class VGGGAP(nn.Module):
    def __init__(self, pretrained=True, num_classes=200):
        super(VGGGAP,self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential((nn.Linear(512,512),nn.ReLU(),nn.Linear(512,4),nn.Sigmoid()))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_classes=200):
        super(VGG16,self).__init__()
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        temp_classifier = torchvision.models.vgg16(pretrained=pretrained).classifier
        removed = list(temp_classifier.children())
        removed = removed[:-1]
        temp_layer = nn.Sequential(nn.Linear(4096,512),nn.ReLU(),nn.Linear(512,4),nn.Sigmoid())
        removed.append(temp_layer)
        self.classifier = nn.Sequential(*removed)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

'''
class InceptionCam(nn.Module):
    def __init__(self, num_classes=1000, large_feature_map=False, **kwargs):
        super(InceptionCam, self).__init__()

        self.large_feature_map = large_feature_map

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # self.SPG_A3_1b = nn.Sequential(
        #     nn.Conv2d(768, 1024, 3, padding=1),
        #     nn.ReLU(True),
        # )
        # self.SPG_A3_2b = nn.Sequential(
        #     nn.Conv2d(1024, 1024, 3, padding=1),
        #     nn.ReLU(True),
        # )
        # self.SPG_A4 = nn.Conv2d(1024, num_classes, 1, padding=0)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.num_classes = num_classes

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)

        x = F.dropout(feat, 0.5, self.training)
        x = self.SPG_A3_1b(x)
        x = F.dropout(x, 0.5, self.training)
        x = self.SPG_A3_2b(x)
        x = F.dropout(x, 0.5, self.training)
                
        feature_map = x
        self.feature_map = feature_map
        
        logits = F.adaptive_avg_pool2d(self.SPG_A4(feature_map), (1, 1))
        pred = logits.view(logits.shape[0:2])
        return pred
'''



def choose_locmodel(model_name,pretrained=False, ckpt_dir=None):
    if model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(2208, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if pretrained:
            model = copy_parameters(model, torch.load('densenet161loc.pth.tar'))
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if pretrained:
            model = copy_parameters(model, torch.load('resnet50loc.pth.tar'))
    elif model_name == 'vgggap':
        model = VGGGAP(pretrained=True,num_classes=1000)
        if pretrained:
            model = copy_parameters(model, torch.load('vgggaploc.pth.tar'))
    elif model_name == 'vgg16':
        model = VGG16(pretrained=True,num_classes=1000)
        if pretrained:
            if ckpt_dir is not None:
                model = copy_parameters(model, torch.load(ckpt_dir))
                print('Done loading loc model')
            else:
                model = copy_parameters(model, torch.load('vgg16loc.pth.tar'))
    elif model_name == 'inceptionv3':
        model = torchvision.models.inception_v3(pretrained=True, aux_logits=True, transform_input=True)
        model.AuxLogits.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        if pretrained:
            if ckpt_dir is not None:
                model = copy_parameters(model, torch.load(ckpt_dir))
                print('Done loading loc model')
            else:
                model = copy_parameters(model, torch.load('inceptionv3loc.pth.tar'))
    else:
        raise ValueError('Do not have this model currently!')
    return model

def choose_clsmodel(model_name,pretrained=False, ckpt_dir=None):
    if model_name == 'vgg16':
        cls_model = torchvision.models.vgg16(pretrained=True)
        cls_model.classifier[6] = nn.Linear(4096, 200, bias=True)
        nn.init.normal_(cls_model.classifier[6].weight, 0, 0.01)
        nn.init.constant_(cls_model.classifier[6].bias, 0)
        
        if ckpt_dir is not None:
            print('begin loading checkpoints')
            cls_model = copy_parameters(cls_model, torch.load(ckpt_dir))
            print('cls model loaded........')
    elif model_name == 'inceptionv3':
        cls_model = torchvision.models.inception_v3(pretrained=True, aux_logits=True, transform_input=True)
        cls_model.fc = nn.Linear(2048, 200, bias=True)
        nn.init.normal_(cls_model.fc.weight, 0, 0.01)
        nn.init.constant_(cls_model.fc.bias, 0)
        if ckpt_dir is not None:
            cls_model = copy_parameters(cls_model, torch.load(ckpt_dir))
            print('cls model loaded........')
    elif model_name == 'resnet50':
        cls_model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'densenet161':
        cls_model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'dpn131':
        cls_model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn131', pretrained=True,test_time_pool=True)
    elif model_name == 'efficientnetb7':
        from efficientnet_pytorch import EfficientNet
        cls_model = EfficientNet.from_pretrained('efficientnet-b7')
    return cls_model


# net = choose_locmodel('inceptionv3', pretrained=False)
# print(net)