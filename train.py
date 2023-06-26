import os
import wandb

os.environ['WANDB_API_KEY'] = ''


def train(training_data, parameters, context):
    wandb.login()

    sweep_id = 'txs1e2qn'
    wandb.agent(sweep_id, entity='epoch-iii', project='CTImageQuality')


if __name__ == '__main__':
    train(None, None, None)
