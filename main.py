from models.resnet import resnet
from models.mobilenet import mobilenet
from models.inception import inception 
#from models.inception import mobilenet
from brevitas.export import export_onnx_qcdq
import torch
import torch.nn as nn

test = False
export = True

def get_model(model_name):
    if model_name == "resnet50":
        return resnet.get_resnet50()
    if model_name == "mobilenetv1":
        return mobilenet.get_mobilenet_v1()
    if model_name == "inceptionv1":
        return inception.get_inception_v1()

model_name = "inceptionv1"
model = get_model(model_name)

if export:
    export_onnx_qcdq(model, input_shape=([1,3,224,224]), export_path=(model_name + "_quant.onnx"), opset_version=13)

if test:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(25):
        print(str(epoch + 1) + "/" + str(25))
        batch_size = 1 
        random_input = torch.randn(batch_size, 3, 224, 224)  # Assuming 3-channel RGB images

        labels = torch.randint(0, 1000, (batch_size,))

        optimizer.zero_grad()

        outputs = model(random_input)
        """
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2].shape)
        """

        loss = criterion(outputs[2], labels)

        loss.backward()
        optimizer.step()

    model.eval()
    for test in range(25):
        print(str(test + 1) + "/" + str(25))
        random_input = torch.randn(batch_size, 3, 224, 224)  # Assuming 3-channel RGB images
        outputs = model(random_input)
