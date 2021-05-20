import torch
import torch.nn as nn

class Net_old(nn.Module):
    def __init__(self):
        super(Net_old, self).__init__()
        self.nets = nn.Sequential(
            torch.nn.Conv2d(1, 2, 3),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(2, 1, 3),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(1, 1, 3)
        )
    def forward(self, x):
        return self.nets(x)

class Net_new(nn.Module):
    def __init__(self):
        super(Net_new, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 3)
        self.r1 = torch.nn.ReLU(True)
        self.conv2 = torch.nn.Conv2d(2, 1, 3)
        self.r2 = torch.nn.ReLU(True)
        self.conv3 = torch.nn.Conv2d(1, 1, 3)
        ##### 在Net_new也加入了一个'nets'属性
        self.nets = nn.Sequential(
            torch.nn.Conv2d(1, 2, 3)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.r1(x)
        x = self.conv2(x)
        x = self.r2(x)
        x = self.conv3(x)
        return x

old_network = Net_old()
torch.save(old_network.cpu().state_dict(), 't.pth')

pretrained_net = torch.load('t.pth')

# Show keys of pretrained model
for key, v in pretrained_net.items():
    print (key)
print('****Before loading********')
new_network = Net_new()
print(torch.sum(old_network.nets[0].weight.data))
print(torch.sum(new_network.conv1.weight.data))
print(torch.sum(new_network.nets[0].weight.data))
for key, _ in new_network.state_dict().items():
    print (key)
print('-----After loading------')
new_network.load_state_dict(pretrained_net, strict=False)
print(torch.sum(old_network.nets[0].weight.data))
print(torch.sum(new_network.conv1.weight.data))
# Hopefully, this value equals to 'old_network.nets[0].weight'
print(torch.sum(new_network.nets[0].weight.data))
for key, _ in new_network.state_dict().items():
    print (key)

from torch.autograd import Variable
import torch.onnx
import torchvision

dummy_input = Variable(torch.randn(10, 3, 224, 224)).cuda()
model = torchvision.models.alexnet(pretrained=True).cuda()
torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)