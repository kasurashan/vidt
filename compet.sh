#!/bin/bash

python3 -m torch.distributed.launch \
 --nproc_per_node=1 \
 --nnodes=1 \
 --use_env coco_competition.py \
 --method vidt \
 --backbone_name swin_base_win7_22k \
 --batch_size 2 \
 --with_box_refine True \
 --coco_path ./datasets/coco2 \
 --resume /workspace/vidts/vidt_test/output/checkpoint.pth \
 --pre_trained none \

