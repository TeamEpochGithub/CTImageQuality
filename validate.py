import os
import os.path as osp
import random
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_warmup as warmup
from scipy.stats import pearsonr, spearmanr, kendalltau
import wandb
from datasets import create_datalists, create_datasets
from models.dncnn import DnCNN
from models.edcnn import EDCNN
from models.get_models import get_model
import output
import output as model_dir


def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def valid(model, test_dataset, best_score, best_score_epoch, epoch, wandb_single_experiment=False, loss_threshold=0.5):

    model.eval()
    total_pred = []
    total_gt = []
    aggregate_results = dict()
    bad_pred_indices = []
    count = 0

    t = tqdm(enumerate(test_dataset), total=len(test_dataset), desc="validation", colour='blue')
    with torch.no_grad():

        # models = {}
        # weight_dir_path = osp.dirname(model_dir.__file__)
        #
        # models['edcnn'] = EDCNN().cuda()
        # models['edcnn'].load_state_dict(torch.load(osp.join(weight_dir_path, 'ED_CNN_epoch_174_alldata.pth')))
        # # self.models['edcnn'].eval()
        # models['dncnn'] = DnCNN().cuda()
        # models['dncnn'].load_state_dict(torch.load(osp.join(weight_dir_path, 'DNCNN_epoch_179_alldata.pth')))
        # # self.models['dncnn'].eval()
        #
        # model_list = list(models.values())
        # model_names = list(models.keys())
        # sum_pred = 0


        for i, (img, label) in t:
            img = img.unsqueeze(0).float()



            # for i, model in enumerate(model_list):
            #     # print(model(x))
            #     pred = model(img.cuda())
            #     print(model_names[i], pred.item())
            #     sum_pred += pred
            # # print(f'sum {sum_pred}')
            #
            # ensemble_pred = (sum_pred / len(model_list))
            # pred = ensemble_pred



            pred = model(img.cuda())
            # pred = torch.full((img.shape[0], 1), random.uniform(1.99, 2.01)).cuda()
            pred_new = pred.cpu().numpy().squeeze(0)

            label_new = label.cpu().numpy()



            print("diff:", pred_new[0] - label_new)
            if abs(pred_new[0] - label_new) > 0.5:
                bad_pred_indices.append(i)
                count += 1
                print(count)
            loss = F.mse_loss(pred.squeeze(), label.cuda())
            # if loss > (loss_threshold ** 2):
            #     print(loss > (loss_threshold ** 2))
            #     bad_pred_indices.append(i)
            print("#")

            # print(round(pred_new[0], 2), label_new)
            total_pred.append(pred_new[0])
            total_gt.append(label_new)
            if i == len(test_dataset) - 1:
                # errors = [abs(x - float(y)) for x, y in zip(total_pred, total_gt)]
                total_pred = np.array(total_pred)
                total_gt = np.array(total_gt)
                aggregate_results["plcc"] = abs(pearsonr(total_pred, total_gt)[0])
                aggregate_results["srocc"] = abs(spearmanr(total_pred, total_gt)[0])
                aggregate_results["krocc"] = abs(kendalltau(total_pred, total_gt)[0])
                aggregate_results["overall"] = abs(pearsonr(total_pred, total_gt)[0]) + abs(
                    spearmanr(total_pred, total_gt)[0]) + abs(kendalltau(total_pred, total_gt)[0])
                std = np.std(total_pred - total_gt)
                aggregate_results["std"] = std
                t.set_postfix({key: round(value, 3) for key, value in aggregate_results.items()})

    aggregate_results['epoch'] = epoch
    if aggregate_results["overall"] > best_score:
        # print("new best model saved")
        best_score = aggregate_results["overall"]
        best_score_epoch = epoch

        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(model.state_dict(), osp.join('output', "model.pth"))
        if wandb_single_experiment:
            wandb.save("model.pth")

    return bad_pred_indices


if __name__ == '__main__':
    configs = {
        'pretrain': 'None',
        'img_size': 512,
        'model': 'Resnet18',
        'epochs': 100,
        'batch_size': 32,
        'weight_decay': 1e-3,
        'lr': 1e-4,
        'min_lr': 0.000006463,
        'RandomHorizontalFlip': True,
        'RandomVerticalFlip': True,
        'RandomRotation': True,
        'ZoomIn': False,
        'ZoomOut': False,
        'use_mix': False,
        'use_avg': False,
        'XShift': False,
        'YShift': False,
        'RandomShear': False,
        'max_shear': 30,  # value in degrees
        'max_shift': 0.5,
        'rotation_angle': 3,
        'zoomin_factor': 0.95,
        'zoomout_factor': 0.05,
    }

    imgs_list, label_list = create_datalists()

    test_dataset, _ = create_datasets(imgs_list, label_list, configs, final_train=True,
                                      patients_out=False, patient_ids_out=[1, 2, 3])
    print(len(test_dataset))

    model = get_model({'model': 'Resnet18', 'pretrain': 'None'})
    model.load_state_dict(
        torch.load(osp.join(osp.dirname(output.__file__), 'Resnet18_epoch_150_1foldout.pth'), map_location="cpu"),
        strict=True)
    model = model.cuda()
    worst_pred_indices = valid(model, test_dataset, 0, 0, 0, False)
