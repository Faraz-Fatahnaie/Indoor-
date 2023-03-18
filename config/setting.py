from pathlib import Path


def setting(CONFIGS):
    BASE_DIR = Path(__file__).resolve().parent

    config = dict()
    config['MIT'] = BASE_DIR.joinpath('data')
    config['N_CHANNEL'] = CONFIGS['dataset']['n_channel']
    config['IMAGE_SIZE'] = CONFIGS['dataset']['image_size']
    config['augmentation'] = CONFIGS['dataset']['augmentation']
    config['NUM_WORKER'] = CONFIGS['dataset']['n_worker']

    config['MODEL_NAME'] = CONFIGS['model']['name']
    config['LR'] = CONFIGS['model']['lr']
    config['LOSS_FUNCTION'] = CONFIGS['model']['loss']
    config['EPOCHS'] = CONFIGS['model']['epoch']
    config['BATCH_SIZE'] = CONFIGS['model']['batch_size']
    config['MIN_LR'] = CONFIGS['model']['min_lr']
    config['PATIENCE'] = CONFIGS['model']['patience']
    config['EARLY_STOP'] = CONFIGS['model']['early_stop']
    config['FACTOR'] = CONFIGS['model']['factor']
    config['SCHEDULER'] = CONFIGS['model']['scheduler']
    config['OPTIMIZER'] = CONFIGS['model']['optim']

    if config['N_CHANNEL'] == 1:
        config['CHANNEL_MODE'] = 'Gray'
    else:
        config['CHANNEL_MODE'] = 'RGB'
    return config
