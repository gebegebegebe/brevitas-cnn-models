from torchvision.models.quantization.resnet import resnet50
from brevitas.export import export_onnx_qcdq
import torch

print(torch.backends.quantized.supported_engines)
model = resnet50(quantize=True)
model_name = "resnet50"
export_onnx_qcdq(model, input_shape=([1,3,224,224]), export_path=(model_name + "_quant.onnx"), opset_version=13)
