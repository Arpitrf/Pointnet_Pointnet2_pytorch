"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

import open3d as o3d
import matplotlib.pyplot as plt

from data_utils.OGDataLoader import SequenceDataset, prepare_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=1, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    collision_count, false_negative, false_positive = 0, 0, 0

    for j, batch in tqdm(enumerate(loader), total=len(loader)):
        points, actions, target = batch['points'], batch['actions'], batch['contacts']
        breakpoint()
        
        # remove later
        votes = 10
        actions_tile = actions.repeat(votes, 1)
        pos_noise = torch.empty(votes, 3).uniform_(-0.005, 0.005)
        actions_tile[:, 3:6] = actions_tile[:, 3:6] + pos_noise

        if not args.use_cpu:
            points, actions, target = points.type(torch.FloatTensor).cuda(), actions.type(torch.FloatTensor).cuda(), target.type(torch.FloatTensor).cuda()
            actions_tile = actions_tile.type(torch.FloatTensor).cuda()
        org_points = points.clone().cpu().numpy()
        points = points.transpose(2, 1)
        # vote_pool = torch.zeros(target.size()[0], num_class).cuda()
        
        # # ---------------------------
        # for _ in range(vote_num):
        #     pred, _ = classifier(points, actions)
        #     # vote_pool += pred
        # # pred = vote_pool / vote_num
        # # pred_choice = pred.data.max(1)[1]

        # probabilities = torch.sigmoid(pred)
        # # Convert probabilities to binary predictions (0 or 1)
        # pred_choice = (probabilities >= 0.5).float()

        # print(f"target: pred_choice, pred_prob: ", target.item(), pred_choice.item(), probabilities.item())

        # # -----------------------------
        
        preds, pred_choices = [], []
        # TODO: Vectorize this
        for i in range(votes):
            pred, _ = classifier(points, actions_tile[i:i+1])
            preds.append(pred)
            probabilities = torch.sigmoid(pred)
            # Changed from 0.5
            pred_choice = (probabilities >= 0.3).float()
            pred_choices.append(pred_choice.item())
    

        count_ones = pred_choices.count(1.0)
        count_zeros = pred_choices.count(0.0)
        if count_ones > count_zeros:
            pred_choice = torch.tensor(1.0).cuda()
            confidence = count_ones
        elif count_zeros > count_ones:
            pred_choice = torch.tensor(0.0).cuda()
            confidence = count_zeros
        else:
            pred_choice = torch.tensor(1.0).cuda()
            confidence = count_ones

        print(f"target: pred_choice, pred_prob: ", target.item(), pred_choice.item(), confidence)
        collision_count += target.item()
        if target.item() == 1.0 and pred_choice.item() == 0.0:
            false_negative += 1
        if target.item() == 0.0 and pred_choice.item() == 1.0:
            false_positive += 1
        print("total collisions, false_negative, false_positive: ", collision_count, false_negative, false_positive)

        # pcd = o3d.geometry.PointCloud()
        # # Assign the points to the PointCloud object
        # pcd.points = o3d.utility.Vector3dVector(org_points[0])
        # # Visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])
        
        # for cat in np.unique(target.cpu()):
        #     classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
        #     class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
        #     class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # data_path = 'data/modelnet40_normal_resampled/'
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    test_dataset = SequenceDataset(
        hdf5_path='/home/arpit/test_projects/OmniGibson/place_in_shelf_data_test2/dataset.hdf5',
        obs_keys=('pcd',),  # observations we want to appear in batches
        # obs_info_keys=('seg_instance_id_info',),
        dataset_keys=(  # can optionally specify more keys here if they should appear in batches
            "actions",
            "grasped",
            "contacts"
        ),
        seq_length=1,  # length-10 temporal sequences
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        hdf5_normalize_obs=False,
        filter_by_attribute=None,  # filter either train or validation data
        image_size=[64, 64],
    )
    testDataLoader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=1,  
        # drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
        collate_fn=prepare_data,
    )

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(args.model)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        print('Test Instance Accuracy: ', instance_acc)


if __name__ == '__main__':
    args = parse_args()
    main(args)
