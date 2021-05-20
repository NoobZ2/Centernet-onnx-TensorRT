import torch
from torch2trt import torch2trt
# create some regular pytorch model...
from torchvision.models import alexnet

model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 512, 512)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(torch.max(torch.abs(y - y_trt)))

#We can save the model as a state_dict.

torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
#We can load the saved model into a TRTModule

from torch2trt import TRTModule

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('alexnet_trt.pth'))