import os

import wandb

os.environ['WANDB_API_KEY'] = '89d2856e8d33911356269aab43c894552d26ba74'

def train(training_data, parameters, context):
    wandb.login()

    sweep_id = ''
    wandb.agent(sweep_id, count=300)
