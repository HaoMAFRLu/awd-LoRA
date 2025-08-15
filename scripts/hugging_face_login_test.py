import os
from loguru import logger

logger.info(f"Setting up Hugging Face environment...")

os.environ.setdefault("HF_HOME", "/lustre/home/hma2/hf")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))

if not os.environ.get("HF_TOKEN"):
    try:
        with open(os.path.join(os.environ["HF_HOME"], "token"), "r") as f:
            os.environ["HF_TOKEN"] = f.read().strip()
    except FileNotFoundError:
        pass

from huggingface_hub import HfApi
try:
    who = HfApi().whoami() 
    logger.info(f"HF login OK. user: {who["name"]}", )
except Exception as e:
    logger.info(f"HF whoami failed: {e}")