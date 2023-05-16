from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
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

def load_efficientnet_model(model_name):
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

    if model_name not in model_dict:
        raise ValueError("Invalid model name. Expected one of: %s" % ", ".join(model_dict.keys()))

    model = model_dict[model_name](pretrained=True)
    model = adapt_efficientnet_to_grayscale(model)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    return model
