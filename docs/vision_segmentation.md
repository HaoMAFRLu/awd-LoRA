# VOC 2012 semantic-segmentation probe

This probe measures spatial information in the frozen SALAAD-Vision
backbones. It restores the final 784 DINO patch tokens to a `28 x 28` grid,
applies one shared `1 x 1` convolution, and bilinearly upsamples the 21 class
logits to `224 x 224`. The CLS token is not used and the backbone remains
frozen.

## Data

Set `VOC2012_ROOT` to the extracted Pascal VOC location. The loader accepts a
path to `VOC2012`, `VOCdevkit`, or the directory containing `VOCdevkit`:

```text
$VOC2012_ROOT/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/Segmentation/
            ├── train.txt
            ├── val.txt
            └── trainval.txt
```

The train pipeline applies a paired random resized crop and horizontal flip.
Validation resizes the shorter side to 256 and takes a paired center crop.
Images use ImageNet normalization; masks use nearest-neighbor interpolation so
class IDs are never blended. VOC's `255` label is ignored.

## Run one backbone

```bash
export VOC2012_ROOT=/path/to/VOCdevkit
export WANDB_API_KEY=...

myenv/bin/python scripts/train_vision.py \
  --config configs/vision_voc2012_teacher_segmentation.yaml
```

The seven committed configs differ only in the selected checkpoint and output
identity:

| Config suffix | Backbone |
| --- | --- |
| `teacher` | official dense DINO ViT-B/8 |
| `vanilla` | vanilla distilled student |
| `salaad_all` | SALAAD on QKV, projection, and MLP matrices |
| `salaad_qkv` | QKV-only SALAAD reconstruction |
| `salaad_qkv_s50_alpha1` | derived S50, alpha 1.0 backbone |
| `salaad_qkv_s50_alpha1p5` | derived S50, alpha 1.5 backbone |
| `salaad_qkv_s50_alpha3` | derived S50, alpha 3.0 backbone |

All filenames follow
`configs/vision_voc2012_<suffix>_segmentation.yaml`. Set
`wandb.enabled: false` for an offline run. Checkpoints are written under
`data/salaad_vision/downstream/voc2012_segmentation/<suffix>/checkpoint.pth`.
Because `model.freeze: true`, the checkpoint stores the task head, optimizer,
history, and config rather than another copy of the backbone.

The configs use `runtime.distributed: auto`. A single process is recommended
for this small pilot. A multi-GPU run can instead launch the same config with
`torch.distributed.run`; `training.batch_size` remains the per-rank batch size.

## Run the cluster sweep

The submit file queues one independent single-GPU job per backbone:

```bash
export VOC2012_ROOT=/cluster/path/to/VOCdevkit
condor_submit_bid <bid> sub/vision_voc2012_segmentation_sweep.sub
```

`getenv = True` passes `VOC2012_ROOT` to the execute node. The submit file does
not download VOC and assumes the seven checkpoint paths already present in the
repository workspace are also available on the cluster.

## Metrics

Each epoch reports cross-entropy loss, mIoU, pixel accuracy, mean class
accuracy, and boundary F1. Metrics exclude label `255` and are accumulated as
additive statistics before distributed reduction. Boundary F1 measures
four-connected semantic-transition localization with the configured pixel
tolerance (`1` by default), computes precision/recall per semantic class, and
reports the mean class F1 over classes with target or predicted boundaries.
