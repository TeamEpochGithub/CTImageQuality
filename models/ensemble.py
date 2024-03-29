from models.edcnn import EDCNN
from models.edcnn2 import EDCNN2
from models.dncnn import DnCNN

import torch
import os.path as osp
import torch.nn as nn
import output as model_dir


class Ensemble(nn.Module):
    def __init__(self, edcnn=True, dncnn=False, edcnn2=False):


        super().__init__()
        self.models = {}
        weight_dir_path = osp.dirname(model_dir.__file__)


        if edcnn:
            self.models['edcnn'] = EDCNN().cuda().eval()
            self.models['edcnn'].load_state_dict(torch.load(osp.join(weight_dir_path, 'ED_CNN_epoch_174_alldata.pth')))
        if dncnn:
            self.models['dncnn'] = DnCNN().cuda().eval()
            self.models['dncnn'].load_state_dict(torch.load(osp.join(weight_dir_path, 'DNCNN_epoch_179_alldata.pth')))
        if edcnn2:
            self.models['edcnn2'] = EDCNN2().cuda().eval()
            self.models['edcnn2'].load_state_dict(torch.load(osp.join(weight_dir_path, 'EDCNN2_epoch_174_alldata.pth')))

    def forward(self, x):
        model_list = list(self.models.values())
        model_names = list(self.models.keys())
        sum_pred = 0
        with torch.no_grad():
            for i, model in enumerate(model_list):
                # print(model(x))
                pred = model(x)
                # print(model_names[i], pred.item())
                sum_pred += pred
            # print(f'sum {sum_pred}')
            # ensemble_pred = sum(model(x) for model in model_list)
            ensemble_pred = (sum_pred / len(model_list))

            # print(f'pred{ensemble_pred}')
        return ensemble_pred





# if __name__ == '__main__':
#     model = Ensemble(edcnn=True, dncnn=True, edcnn2=True).cuda()
#
#     x_image = torch.randn(1, 512, 512)
#     pred = model(x_image.cuda())
#     print(pred)
