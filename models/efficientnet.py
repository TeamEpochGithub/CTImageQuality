from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, \
    efficientnet_b5, efficientnet_b6, efficientnet_b7, EfficientNet_B0_Weights, EfficientNet_B1_Weights, \
    EfficientNet_B2_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, \
    EfficientNet_B6_Weights, EfficientNet_B7_Weights
import torch.nn as nn


def adapt_efficientnet_to_grayscale(model):
    num_channels = 1  # Grayscale images have 1 channel
    out_channels = model.features[0][0].out_channels
    kernel_size = model.features[0][0].kernel_size
    stride = model.features[0][0].stride
    padding = model.features[0][0].padding
    model.features[0][0] = nn.Conv2d(num_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=False)
    return model


def load_efficientnet_model(model_name, pretrained_weights):
    model_dict = {
        'b0': efficientnet_b0,
        'b1': efficientnet_b1,
        'b2': efficientnet_b2,
        'b3': efficientnet_b3,
        'b4': efficientnet_b4,
        'b5': efficientnet_b5,
        'b6': efficientnet_b6,
        'b7': efficientnet_b7,
    }

    default_weights_dict = {
        'b0': EfficientNet_B0_Weights.DEFAULT,
        'b1': EfficientNet_B1_Weights.DEFAULT,
        'b2': EfficientNet_B2_Weights.DEFAULT,
        'b3': EfficientNet_B3_Weights.DEFAULT,
        'b4': EfficientNet_B4_Weights.DEFAULT,
        'b5': EfficientNet_B5_Weights.DEFAULT,
        'b6': EfficientNet_B6_Weights.DEFAULT,
        'b7': EfficientNet_B7_Weights.DEFAULT,
    }

    pretrained_weights_dict = {  # This is now the same as default, but we should load in other weights here
        'b0': EfficientNet_B0_Weights.IMAGENET1K_V1,
        'b1': EfficientNet_B1_Weights.IMAGENET1K_V1,
        'b2': EfficientNet_B2_Weights.IMAGENET1K_V1,
        'b3': EfficientNet_B3_Weights.IMAGENET1K_V1,
        'b4': EfficientNet_B4_Weights.IMAGENET1K_V1,
        'b5': EfficientNet_B5_Weights.IMAGENET1K_V1,
        'b6': EfficientNet_B6_Weights.IMAGENET1K_V1,
        'b7': EfficientNet_B7_Weights.IMAGENET1K_V1,
    }

    if model_name not in model_dict:
        raise ValueError("Invalid model name. Expected one of: %s" % ", ".join(model_dict.keys()))

    if not pretrained_weights:
        model = model_dict[model_name](weights=default_weights_dict[model_name])
    else:
        model = model_dict[model_name](weights=pretrained_weights_dict[model_name])

    model = adapt_efficientnet_to_grayscale(model)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    return model
