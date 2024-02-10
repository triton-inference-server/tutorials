import argparse
import torch
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument("--opset", type=int, default=14, help="ONNX opset version to generate models with.")
args = parser.parse_args()

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.resnet50(pretrained=True).cuda()

# Fixed Shape
torch.onnx.export(model, dummy_input, "resnet50_fixed.onnx", verbose=True, opset_version=args.opset)
