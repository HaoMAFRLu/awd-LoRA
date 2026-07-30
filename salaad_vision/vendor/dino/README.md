# Vendored DINO model definition

`vision_transformer.py` comes from the archived Facebook Research DINO
repository at commit
[`7c446df5b9f45747937fb0d72314eb9f7b66930a`](https://github.com/facebookresearch/dino/commit/7c446df5b9f45747937fb0d72314eb9f7b66930a).
It is distributed under the Apache License 2.0 included in this directory.

Local modification: the original `from utils import trunc_normal_` statement
was replaced by `from torch.nn.init import trunc_normal_`. This prevents an
ambiguous import of awd-LoRA's unrelated top-level `utils` package. It does
not alter the ViT-B/8 architecture or its state-dict layout.
