"""This script is used to show that the pretrained llm models do not
have the low-rank + sparse structure.
"""
import sys, os
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model
from salad.static_rpca import StaticRPCA

hf_login_once()
ROOT = get_parent_path(lvl=1)

def main(MODEL_NAME: str) -> None:
    path_folder = os.path.join(ROOT, 'data', 'pretrained_recover', MODEL_NAME.replace('/', '_'))
    mkdir(path_folder)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16, 
        device_map='cpu'
    )

    static_rpca = StaticRPCA(model, path_folder)
    static_rpca.recover_X()
    static_rpca.destroy()

if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    main(MODEL_NAME)