import os
import sys
from collections import defaultdict
from typing import Dict, Optional

import torch
from loguru import logger
from torch.types import Number

from df.utils import get_branch_name, get_commit_hash, get_device, get_host

_logger_initialized = False


def init_logger(file: Optional[str] = None, level: str = "INFO"):
    global _logger_initialized
    if _logger_initialized:
        logger.debug("Logger already initialized.")
        return
    logger.remove()
    level = level.upper()
    if level != "NONE":
        log_format = get_log_format(debug=level == "DEBUG")
        logger.add(sys.stdout, level=level, format=log_format)
        if file is not None:
            logger.add(file, level=level, format=log_format)

        logger.info(f"Running on torch {torch.__version__}")
        logger.info(f"Running on host {get_host()}")
        commit = get_commit_hash()
        if commit is not None:
            logger.info(f"Git commit: {commit}, branch: {get_branch_name()}")
        if (jobid := os.getenv("SLURM_JOB_ID")) is not None:
            logger.info(f"Slurm jobid: {jobid}")
    _logger_initialized = True


def get_log_format(debug=False):
    if debug:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            " | <level>{level: <8}</level>"
            " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
            " | <level>{message}</level>"
        )
    else:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            " | <level>{level: <8}</level>"
            " | <cyan>DF</cyan>"
            " | <level>{message}</level>"
        )


def log_metrics(prefix: str, metrics: Dict[str, Number]):
    msg = prefix
    stages = defaultdict(str)
    for n, v in sorted(metrics.items()):
        m = f" | {n}: {v:.5g}"
        if "stage" in n:
            s = n.split("stage_")[1].split("_snr")[0]
            stages[s] += m.replace(f"stage_{s}_", "")
        else:
            msg += m
    for s, msg in stages.items():
        logger.info(f"{prefix} | stage {s}" + msg)
    logger.info(msg)


def log_model_summary(model: torch.nn.Module, verbose=False):
    import ptflops

    from df.model import ModelParams

    # Generate input of 1 second audio
    # Necessary inputs are:
    #   spec: [B, 1, T, F, 2], F: freq bin
    #   feat_erb: [B, 1, T, E], E: ERB bands
    #   feat_spec: [B, 2, T, C*2], C: Complex features
    p = ModelParams()
    b = 1
    t = p.sr // p.hop_size
    device = get_device()
    spec = torch.randn([b, 1, t, p.fft_size // 2 + 1, 2]).to(device)
    feat_erb = torch.randn([b, 1, t, p.nb_erb]).to(device)
    feat_spec = torch.randn([b, 1, t, p.nb_df, 2]).to(device)

    macs, params = ptflops.get_model_complexity_info(
        model,
        (t,),
        input_constructor=lambda _: {"spec": spec, "feat_erb": feat_erb, "feat_spec": feat_spec},
        as_strings=False,
        print_per_layer_stat=verbose,
        verbose=verbose,
    )
    logger.info(f"Model complexity: {params/1e6:.3f}M #Params, {macs/1e6:.1f}M MACS")
