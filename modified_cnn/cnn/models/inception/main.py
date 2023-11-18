import torch
from inception import googlenet
from brevitas.export import export_onnx_qcdq

model_name = "inception_v1"
model = googlenet()
#export_onnx_qcdq(model, input_shape=([1,3,224,224]), export_path=(model_name + "_quant.onnx"), opset_version=13)

image = torch.rand(1, 3, 224, 224)
model.eval()
#model.train()
outputs = model(image)
for output in outputs:
    print(type(output))
