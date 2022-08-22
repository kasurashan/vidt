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

from tqdm import tqdm


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
    #dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    #print("# train:", len(dataset_train), ", # val", len(dataset_val))


    dataset_val_original = coco.build_original(image_set='val', args=args)

    # data samplers
    if args.distributed:
        #sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        #sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    #batch_sampler_train = torch.utils.data.BatchSampler(
        #sampler_train, args.batch_size, drop_last=True)

    #data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   #collate_fn=utils.collate_fn, num_workers=args.num_workers)
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

    
    
    allList = []
    #test_iter_original0 = iter(dataset_val_original)
    cnt = 0
    for test_iter_original in tqdm(dataset_val_original):
        
        image_id = int(test_iter_original[1]['image_id'].numpy()[0])

        tf = transforms.ToTensor()
        test_iter_num = tf(test_iter_original[0]).unsqueeze(0)
        test_iter_info = test_iter_original[1]

        output =  model(test_iter_num)
    

        output_final = postprocessors['bbox'](output, test_iter_info["orig_size"].unsqueeze(0).to(device))
        #print(output_final)
        

        scores = output_final[0]['scores'].tolist()
        category_id = output_final[0]['labels'].tolist()
        boxes = output_final[0]['boxes'].tolist() #xmin, ymin, xmax, ymax
        
        dict = {}
        for idx, box in enumerate(boxes):
            #cnt += 1
            dict =  {'image_id':image_id, \
                     'category_id': category_id[idx], \
                     'bbox': [box[0], box[1], box[2]-box[0], box[3]-box[1] ], \
                     'score':scores[idx]}
            
            allList.append(dict)
        
    #print(cnt)

    file_path = "./test.json"

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(allList, file)


    
        




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


