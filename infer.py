import os
import datetime
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import resource
import datasets
import util.misc as utils
from datasets import build_dataset
from methods import build_model
from arguments import get_args_parser
import argparse
import matplotlib.pyplot as plt
from datasets import coco  #######
import torchvision.transforms as transforms
from engine import evaluate, train_one_epoch, train_one_epoch_with_teacher
from datasets import build_dataset, get_coco_api_from_dataset



def main(args):

    # distributed data parallel setup
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # import pdb;pdb.set_trace()
    model, criterion, postprocessors = build_model(args)
    model.eval()
    model.to(device)

    # parallel model setup
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module


    # print parameter info.
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    # build data loader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    print("# train:", len(dataset_train), ", # val", len(dataset_val))


    dataset_val_original = coco.build_original(image_set='val', args=args)

    # data samplers
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)



    # resume from a checkpoint or eval with a checkpoint
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print('load a checkpoint from', args.resume)

    
    
    
    test_iter_original0 = iter(dataset_val_original)
    
    for i in range(16):
        test_iter_original = next(test_iter_original0)
        
        gt_bbox = test_iter_original[1]['boxes']

        tf = transforms.ToTensor()
        test_iter_num = tf(test_iter_original[0]).unsqueeze(0)
        test_iter_info = test_iter_original[1]

        output =  model(test_iter_num)
    
        #probas = output['pred_logits'].sigmoid()[0, :, :-1]
        #keep = probas.max(-1).values > 0.2
        #vis_indexs = torch.nonzero(keep).squeeze(1)

    
        #print(output)
        output_final = postprocessors['bbox'](output, test_iter_info["orig_size"].unsqueeze(0).to(device))
        #print(output_final)
        scores = output_final[0]['scores']
        boxes = output_final[0]['boxes'] 
        labels = output_final[0]['labels']
        LMS = output_final[0]['LMS']  #####
       
        plot_results(test_iter_original[0], gt_bbox, scores, boxes, labels, LMS, i) #####






#coco classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0,0,0]]


def plot_results(pil_img, gt, prob, boxes, labels, LMS, i): #####
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100


    for p, (xmin, ymin, xmax, ymax), (xmin_gt, ymin_gt, xmax_gt, ymax_gt), l, lms in zip(prob, boxes.tolist(), gt.tolist(), labels,  LMS):######
        
        ax.add_patch(plt.Rectangle((xmin_gt, ymin_gt),  xmax_gt - xmin_gt, ymax_gt - ymin_gt,
                                   fill=False, color=colors[5], linewidth=5))

        if lms == 'small':
            ax.add_patch(plt.Rectangle((xmin, ymin),  xmax - xmin, ymax - ymin,
                                   fill=False, color=colors[0], linewidth=3))

        if lms == 'medium':
            ax.add_patch(plt.Rectangle((xmin, ymin),  xmax - xmin, ymax - ymin,
                                   fill=False, color=colors[1], linewidth=3))

        if lms == 'large':
            ax.add_patch(plt.Rectangle((xmin, ymin),  xmax - xmin, ymax - ymin,
                                   fill=False, color=colors[3], linewidth=3))

        
        text = f'{CLASSES[l]}: {p:0.2f}: {lms}'#####
        ax.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5))
        
        if p < 0.23 :
            break


    plt.axis('off')
    plt.savefig(f'savefig_default{i}.png')





if __name__ == '__main__':

    parser = argparse.ArgumentParser('ViDT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()


    # set dim_feedforward differently
    # standard Transformers use 2048, while Deformable Transformers use 1024
    if args.method == 'vidt_wo_neck':
        args.dim_feedforward = 2048
    else:
        args.dim_feedforward = 1024

    main(args)


