import yaml
import os
import logging
from datetime import datetime

import torch.distributed as dist

def load_config(config_path):
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Setup logging configuration."""
    base_output_dir = config["training"]["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    
    # Update config for output_dir
    config["training"]["output_dir"] = output_dir

    # Ensure the output directory exists before writing the log file
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler()
        ],
        force=True  # override any previous logging configuration
    )

    logging.info("Logging initialised – saving to %s", log_path)
    return output_dir


def is_rank_zero():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
