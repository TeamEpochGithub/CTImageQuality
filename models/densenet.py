from torchvision.models import densenet121, densenet161, densenet169, densenet201, DenseNet121_Weights, \
    DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights
import torch.nn as nn


def adapt_densenet_to_grayscale(model):
    num_channels = 1  # Grayscale images have 1 channel
    model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def load_densenet_model(model_name, pretrained_weights, out_channel=1, dropout_prob=0.5):
    model_dict = {
        '121': densenet121,
        '161': densenet161,
        '169': densenet169,
        '201': densenet201,
    }

    default_weights_dict = {
        '121': DenseNet121_Weights.DEFAULT,
        '161': DenseNet161_Weights.DEFAULT,
        '169': DenseNet169_Weights.DEFAULT,
        '201': DenseNet201_Weights.DEFAULT,
    }

    if model_name not in model_dict:
        raise ValueError("Invalid model name. Expected one of: %s" % ", ".join(model_dict.keys()))

    model = model_dict[model_name](pretrained=default_weights_dict[model_name])

    model = adapt_densenet_to_grayscale(model)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_prob),
        nn.Linear(num_features, out_channel)
    )

    return model
