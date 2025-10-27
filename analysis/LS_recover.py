"""This script is used to show that the RPCA algorithm can recover low-rank and sparse components 
from the sum of low-rank matrices L and sparse matrices S.
"""
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *
from salad.register import get_model
from salad.static_rpca import StaticRPCA

hf_login_once()
ROOT = get_parent_path(lvl=1)

def main(MODEL_TYPE: str, FILE: str) -> None:

    path_folder = os.path.join(ROOT, 'data', 'salad', MODEL_TYPE, FILE)
    path_cfg_model = os.path.join(path_folder, MODEL_TYPE+'_model.json')

    model = get_model(path_cfg_model)
    # load the original model weights X
    load_model(model, os.path.join(path_folder, 'model.pth'))

    static_rpca = StaticRPCA(model, path_folder)
    static_rpca.recover_X()
    static_rpca.recover_LS()
    static_rpca.destroy()

if __name__ == "__main__":
    MODEL_TYPE = 'llama_1b'
    FILE = '20251016_233939' 
    main(MODEL_TYPE, FILE)