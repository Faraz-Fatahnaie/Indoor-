from pathlib import Path
import json


def setting(config_file=None):
    BASE_DIR = Path(__file__).resolve().parent.parent

    if config_file is None:
        config_name = 'configs'  # config file name in config dir
        config_dir = BASE_DIR.joinpath('config')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)

    config = dict()
    config['MIT'] = BASE_DIR.joinpath('data')
    config['N_CHANNEL'] = config_file['dataset']['n_channel']
    config['IMAGE_SIZE'] = config_file['dataset']['image_size']
    config['AUGMENTATION'] = config_file['dataset']['augmentation']
    config['NUM_WORKER'] = config_file['dataset']['n_worker']

    config['MODEL_NAME'] = config_file['model']['name']

    config['LOSS_FUNCTION'] = config_file['model']['loss']
    config['EPOCHS'] = config_file['model']['epoch']
    config['BATCH_SIZE'] = config_file['model']['batch_size']
    config['OPTIMIZER'] = config_file['model']['optim']
    config['LR'] = config_file['model']['lr']
    config['WEIGHT_DECAY'] = config_file['model']['weight_decay']
    config['SCHEDULER'] = config_file['model']['scheduler']
    config['MIN_LR'] = config_file['model']['min_lr']
    config['PATIENCE'] = config_file['model']['patience']
    config['EARLY_STOP'] = config_file['model']['early_stop']
    config['FACTOR'] = config_file['model']['factor']
    config['GAMMA'] = config_file['model']['gamma']
    config['REGULARIZATION'] = config_file['model']['regularization']


    if config['N_CHANNEL'] == 1:
        config['CHANNEL_MODE'] = 'Gray'
    else:
        config['CHANNEL_MODE'] = 'RGB'

    return config, config_file
