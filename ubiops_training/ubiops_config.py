import ubiops

configuration = ubiops.Configuration()
# Configure API token authorization
configuration.api_key['Authorization'] = "Token c27d7e09d942f5f9f7f1b7120a48a51b3e832525"

api_client = ubiops.ApiClient(configuration)
training_instance = ubiops.Training(api_client)

project_name = 'default-project-team-epoch'  # str
experiment_name = 'ct-image-quality'  # str
run_name = 'run_name_example'  # str

new_run = training_instance.experiment_runs_create(
    project_name=project_name,
    experiment_name=experiment_name,
    data=ubiops.ExperimentRunCreate(
        name=run_name,
        description='Trying out a run',
        training_code='../pretrain/train.py',
        training_data='ubiops-file://default/npy_imgs.zip',
        parameters={
            'nr_epochs': 15,  # example given_params
            'batch_size': 32
        },
        timeout=14400
    )
)
