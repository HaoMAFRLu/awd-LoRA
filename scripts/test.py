"""Test for model, dataloader, and tokenizer.
"""

import argparse, math, torch, transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import dataloaders


transformers.logging.set_verbosity_error()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def main():
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=256)
    
    cfg = {
            "architectures": [
                "LLaMAForCausalLM"
            ],
            "bos_token_id": 0,
            "eos_token_id": 1,
            "hidden_act": "silu",
            "hidden_size": 512,
            "intermediate_size": 1376,
            "initializer_range": 0.02,
            "max_sequence_length": 1024,
            "model_type": "llama",
            "num_attention_heads": 8,
            "num_hidden_layers": 8,
            "pad_token_id": -1,
            "rms_norm_eps": 1e-06,
            "transformers_version": "4.28.1",
            "use_cache": True,
            "vocab_size": 32000
        }
    model_config = AutoConfig.from_pretrained(cfg)
    model = LlamaForCausalLM(model_config)
    data = dataloaders.load_dataset("allenai/c4", "en", split="train", streaming=True)
    seed_for_shuffle = 42
    # data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    # dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

if __name__ == "__main__":
    main()