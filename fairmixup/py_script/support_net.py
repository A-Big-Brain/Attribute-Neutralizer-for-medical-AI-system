import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class convnext_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnext = torchvision.models.convnext_small(weights='IMAGENET1K_V1')
        self.convnext.classifier = Identity()

    def forward(self, x):
        outputs = self.convnext(x)
        return outputs.view(-1, 768)

class LinearModel(nn.Module):
    def __init__(self, c_n):
        super(LinearModel, self).__init__()
        self.nor = nn.LayerNorm(768)
        self.liner = nn.Linear(768, c_n)

    def forward(self, x):
        x = self.nor(x)
        x = self.liner(x)
        return x

def get_transforms():
    transforms = torch.nn.Sequential(
        #tt.Resize(size=600),
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomRotation(degrees=(-10, 10), expand=False),
        tt.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        tt.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.12), scale=(0.9, 0.99)),
        tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    return transforms
