from training.utils import create_default_args

lwf_args = create_default_args({
    'cuda': 1,
    'alpha': 1,
    'temperature': 2,
    'train_epochs': 150,
    'learning_rate': 1e-3,
    'train_mb_size': 512,
    'seed': None,
    'dataset_root': None,
    'save_folder': 'lwf_saved_models/'
})


mas_args = create_default_args({
    'cuda': 0,
    'alpha': 0.5,
    'lambda_reg': 1.,
    'verbose': True,
    'learning_rate': 1e-3,
    'train_epochs': 150,
    'train_mb_size': 512,
    'seed': None,
    'dataset_root': None,
    'save_folder': 'mas_saved_models/'
})
