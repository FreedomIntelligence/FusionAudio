import torch
import numpy as np
from torch.serialization import add_safe_globals
import argparse
import yaml  # 使用 PyYAML
from tqdm import tqdm
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from models.ase_model import ASE
import torch.distributed as dist
from pretrain import validate
from loguru import logger
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_seed,
    AverageMeter, t2a, a2t, set_logger, log_results,
)

# 添加允许的全局对象
add_safe_globals([np.core.multiarray.scalar])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="settings/inference.yaml", type=str,
                        help="Setting files")

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # setup distribution mode
    device = torch.device(config["device"])

    # setup seed
    seed = config["seed"]
    setup_seed(seed)

    dataset_name = config["data_args"]["dataset"]

    # load evaluation datamodule
    datamodule = AudioCaptionDataModule(config, dataset_name)
    test_loader = datamodule.test_dataloader()

    # setup model
    model = ASE(config)
    model = model.to(device)

    main_logger = logger.bind(indent=1)

    from pprint import PrettyPrinter
    printer = PrettyPrinter()
    main_logger.info('Eval:\n'
                     f'{printer.pformat(config)}')

    model = ASE(config)
    model.to(device)

    ckpt_path = config['ckpt_path']
    cp = torch.load(ckpt_path, weights_only=False)  # 修改此处
    model.load_state_dict(cp['model'], strict=False)
    model.eval()
    main_logger.info(f"Loaded weights from {config['ckpt_path']}")
    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')

    # eval
    metrics = validate(model, test_loader, device)

    main_logger.info('###### Eval on {} done ######'.format(dataset_name))
    main_logger.info('###### Best Metrics {} ######'.format(metrics))

if __name__ == '__main__':
    main()