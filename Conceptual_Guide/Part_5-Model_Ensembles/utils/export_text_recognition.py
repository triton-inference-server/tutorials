import torch
from model import STRModel
from pathlib import Path

# Create PyTorch Model Object
model = STRModel(input_channels=1, output_channels=512, num_classes=37)

# Load model weights from external file
state = torch.load("downloads/None-ResNet-None-CTC.pth")
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

# Create ONNX file by tracing model
model_directory = Path("model_repository/text_recognition/1/")
model_directory.mkdir(parents=True, exist_ok=True)
trace_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(
    model,
    trace_input,
    model_directory / "model.onnx",
    verbose=True,
    dynamic_axes={"input.1": [0], "308": [0]},
)
