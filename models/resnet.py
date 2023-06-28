from torchvision.models import resnet18, resnet34, resnet152, resnet50, resnet101, ResNet18_Weights, ResNet34_Weights, \
    ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import torch.nn as nn


def adapt_resnet_to_grayscale(model):
    num_channels = 1  # Grayscale images have 1 channel
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def load_resnet_model(model_name, pretrained_weights, out_channel=1):
    model_dict = {
        '18': resnet18,
        '34': resnet34,
        '50': resnet50,
        '101': resnet101,
        '152': resnet152,
    }

    default_weights_dict = {
        '18': ResNet18_Weights.DEFAULT,
        '34': ResNet34_Weights.DEFAULT,
        '50': ResNet50_Weights.DEFAULT,
        '101': ResNet101_Weights.DEFAULT,
        '152': ResNet152_Weights.DEFAULT,
    }

    pretrained_weights_dict = {  # This is now the same as default, but we should load in other weights here
        '18': ResNet18_Weights.IMAGENET1K_V1,
        '34': ResNet34_Weights.IMAGENET1K_V1,
        '50': ResNet50_Weights.IMAGENET1K_V1,
        '101': ResNet101_Weights.IMAGENET1K_V1,
        '152': ResNet152_Weights.IMAGENET1K_V1,
    }

    if model_name not in model_dict:
        raise ValueError("Invalid model name. Expected one of: %s" % ", ".join(model_dict.keys()))

    # if not pretrained_weights:
    #     model = model_dict[model_name](weights=default_weights_dict[model_name])
    # else:
    #     model = model_dict[model_name](weights=pretrained_weights_dict[model_name])
    model = model_dict[model_name]()
    model = adapt_resnet_to_grayscale(model)
    model.fc = nn.Linear(model.fc.in_features, out_channel)

    return model
