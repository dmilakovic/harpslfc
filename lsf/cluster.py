#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:34:48 2026

@author: dmilakov

harps/lsf/cluster.py
Ray cluster initialisation for interactive and SLURM environments.
"""
import os
import ray
import logging
logger = logging.getLogger(__name__)

def get_jax_platform() -> str:
    """
    Detect the best available JAX backend.
    Returns 'gpu', 'tpu', or 'cpu' depending on what's available.
    """
    import jax
    for backend in ('gpu', 'tpu', 'cpu'):
        try:
            devices = jax.devices(backend)
            if devices:
                logger.info(f"JAX backend selected: {backend} "
                            f"({len(devices)} device(s) found)")
                return backend
        except RuntimeError:
            continue
    return 'cpu'  # always available as fallback

def init_ray(num_gpus_per_node: int | None = None) -> None:
    """
    Initialise Ray for the current environment.
    
    - Inside a SLURM job: connects all allocated nodes into one Ray cluster.
    - Outside SLURM: starts a local Ray instance using all available GPUs.
    
    Always call this once at the top of your pipeline script before any
    ray.remote calls.
    """
    if ray.is_initialized():
        return

    platform = get_jax_platform()
    runtime_env = {
        "env_vars": {
            # Prevent JAX from preallocating all GPU memory — essential when
            # multiple Ray workers share a node's GPUs.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            # Silence TPU warnings — we are on GPU-only nodes.
            "JAX_PLATFORMS": platform,
        }
    }

    if "SLURM_JOB_ID" in os.environ:
        _init_ray_slurm(runtime_env, num_gpus_per_node)
    else:
        _init_ray_local(runtime_env)


def _init_ray_local(runtime_env: dict) -> None:
    import jax
    platform = get_jax_platform()
    n_gpu    = len(jax.devices(platform)) if platform == 'gpu' else 0
    logger.info(f"Starting local Ray instance with {platform} platform "
                f"({n_gpu} GPU(s)).")
    ray.init(num_gpus=n_gpu, runtime_env=runtime_env)


def _init_ray_slurm(runtime_env: dict, num_gpus_per_node: int | None) -> None:
    """
    Launch Ray head on SLURM node 0, workers on all other nodes.
    Expects the SLURM script to call this function on every task.
    """
    import subprocess
    import socket
    import time

    node_id   = int(os.environ.get("SLURM_NODEID", 0))
    n_gpus    = num_gpus_per_node or int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    head_node = os.environ["SLURM_NODELIST"].split(",")[0].split("[")[0]

    # Resolve head node IP
    head_ip = socket.gethostbyname(head_node)
    port    = 6379

    if node_id == 0:
        logger.info(f"Starting Ray head node on {head_ip}:{port}")
        ray.init(
            address          = "auto",
            num_gpus         = n_gpus,
            runtime_env      = runtime_env,
            include_dashboard= False,
        )
    else:
        time.sleep(5)  # give head node time to start
        logger.info(f"Node {node_id} connecting to Ray head at {head_ip}:{port}")
        ray.init(
            address     = f"{head_ip}:{port}",
            num_gpus    = n_gpus,
            runtime_env = runtime_env,
        )

    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")


def get_num_gpus() -> int:
    """Return total number of GPUs visible to the Ray cluster."""
    resources = ray.cluster_resources()
    return int(resources.get("GPU", 0))