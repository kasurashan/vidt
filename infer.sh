#!/bin/bash

python3 -m torch.distributed.launch \
 --nproc_per_node=1 \
 --nnodes=1 \
 --use_env infer.py \
 --method vidt \
 --backbone_name swin_nano \
 --batch_size 2 \
 --with_box_refine True \
 --coco_path ./datasets/cocosmall \
 --resume /workspace/vidt_test/eval_nano_50_25000_333334/checkpoint.pth \
 --pre_trained none \