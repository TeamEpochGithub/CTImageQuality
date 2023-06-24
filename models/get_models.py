from models.densenet import load_densenet_model
from models.dncnn import DnCNN
from models.edcnn2 import EDCNN2
from models.efficient_swin import Efficientnet_Swin
from models.efficient_swinv2 import Efficientnet_Swinv2
from models.efficientnet import load_efficientnet_model
from models.res34_swin import Resnet34_Swin
from models.res34_swinv2 import Resnet34_Swinv2
from models.resnet import load_resnet_model
from models.edcnn import EDCNN
from models.unet import UNet


def get_model(configs):
    models = {'Resnet18': load_resnet_model('18', configs['pretrain']),
              'Resnet34': load_resnet_model('34', configs['pretrain']),
              'Resnet50': load_resnet_model('50', configs['pretrain']),
              'Resnet101': load_resnet_model('101', configs['pretrain']),
              'Resnet152': load_resnet_model('152', configs['pretrain']),
              'Efficientnet_B0': load_efficientnet_model('b0', configs['pretrain']),
              'Efficientnet_B1': load_efficientnet_model('b1', configs['pretrain']),
              'Efficientnet_B2': load_efficientnet_model('b2', configs['pretrain']),
              'Efficientnet_B3': load_efficientnet_model('b3', configs['pretrain']),
              'Efficientnet_B4': load_efficientnet_model('b4', configs['pretrain']),
              'Efficientnet_B5': load_efficientnet_model('b5', configs['pretrain']),
              'Efficientnet_B6': load_efficientnet_model('b6', configs['pretrain']),
              'Efficientnet_B7': load_efficientnet_model('b7', configs['pretrain']),
              'Efficientnet_Swin': Efficientnet_Swin, 'Efficientnet_Swinv2': Efficientnet_Swinv2,
              'Resnet34_Swin': Resnet34_Swin, 'Resnet34_Swinv2': Resnet34_Swinv2,
              'ED_CNN': EDCNN(), 'EDCNN2': EDCNN2(),
              'Densenet121': load_densenet_model('121', configs['pretrain']),
              'DNCNN': DnCNN(), 'UNET': UNet()}

    model = models[configs['model']]
    return model
