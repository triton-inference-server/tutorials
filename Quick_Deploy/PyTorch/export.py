import torch
import torch_tensorrt
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval().to("cuda")
torch.save(model, "model.pt")
