from torchvision.models import resnet18, resnet34, resnet152, resnet50, resnet101
import torch.nn as nn
def adapt_resnet_to_grayscale(model):
    num_channels = 1  # Grayscale images have 1 channel
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def load_resnet_model(model_name):
    model_dict = {
        '18': resnet18,
        '34': resnet34,
        '50': resnet50,
        '101': resnet101,
        '152': resnet152,
    }

    if model_name not in model_dict:
        raise ValueError("Invalid model name. Expected one of: %s" % ", ".join(model_dict.keys()))

    model = model_dict[model_name](pretrained=True)
    model = adapt_resnet_to_grayscale(model)
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
