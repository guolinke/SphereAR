import logging

import torch
import torch.distributed as dist
from torch.nn import functional as F


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_ps = []
    ps = []

    for e, m in zip(ema_model.parameters(), model.parameters()):
        if m.requires_grad:
            ema_ps.append(e)
            ps.append(m)
    torch._foreach_lerp_(ema_ps, ps, 1.0 - decay)


@torch.no_grad()
def sync_frozen_params_once(ema_model, model):
    for e, m in zip(ema_model.parameters(), model.parameters()):
        if not m.requires_grad:
            e.copy_(m)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
