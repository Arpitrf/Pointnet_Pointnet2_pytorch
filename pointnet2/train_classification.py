"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse
import wandb

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.OGDataLoader import SequenceDataset, prepare_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=1, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='use wandb')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, batch in tqdm(enumerate(loader), total=len(loader)):
        # points, target = batch 
        # actions = torch.tensor([])
        points, actions, target = batch['points'], batch['actions'], batch['contacts'] 

        if not args.use_cpu:
            points, actions, target = points.type(torch.FloatTensor).cuda(), actions.type(torch.FloatTensor).cuda(), target.type(torch.FloatTensor).cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points, actions)
        # pred_choice = pred.data.max(1)[1]
        probabilities = torch.sigmoid(pred)
        # Convert probabilities to binary predictions (0 or 1)
        pred_choice = (probabilities >= 0.5).float()

        # for cat in np.unique(target.cpu()):
        #     # classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
        #     classacc = pred_choice[target == cat].eq(target[target == cat].data).cpu().sum()
        #     class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
        #     class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    wandb.log({"test/correct": instance_acc})

    return instance_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    wandb.init(
        project="safety_violation_detection",
        reinit=True,
        mode="online" if args.use_wandb else "offline",
        settings=wandb.Settings(start_method="fork"),
    )

    # train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    # loading custom dataset
    train_dataset = SequenceDataset(
        hdf5_path='/home/arpit/test_projects/OmniGibson/place_in_shelf_data/dataset.hdf5',
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
        filter_by_attribute='train',  # filter either train or validation data
        image_size=[64, 64],
    )
    trainDataLoader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=10,  
        # drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
        collate_fn=prepare_data,
    )

    test_dataset = SequenceDataset(
        hdf5_path='/home/arpit/test_projects/OmniGibson/place_in_shelf_data/dataset.hdf5',
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
        filter_by_attribute='valid',  # filter either train or validation data
        image_size=[64, 64],
    )
    testDataLoader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=10,  
        # drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
        collate_fn=prepare_data,
    )


    '''MODEL LOADING'''
    num_class = args.num_category
    print("args.model: ", args.model)
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, batch in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # points, target = batch 
            # actions = torch.tensor([])
            points, actions, target = batch['points'], batch['actions'], batch['contacts'] 
            # print("point, actions, target: ", points.shape, actions.shape, target.shape)
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            # Commented by Arpit
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # Added by Arpit
            points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3], sigma=0.001, clip=0.005)

            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, actions, target = points.type(torch.FloatTensor).cuda(), actions.type(torch.FloatTensor).cuda(), target.type(torch.FloatTensor).cuda()

            pred, trans_feat = classifier(points, actions)
            # loss = criterion(pred, target.long(), trans_feat)
            loss = criterion(pred, target, trans_feat)
            pred_choice = pred.data.max(1)[1]
            
            probabilities = torch.sigmoid(pred)
            # Convert probabilities to binary predictions (0 or 1)
            pred_choice = (probabilities >= 0.5).float()

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

            wandb.log({"train/loss": loss})
            wandb.log({"train/correct": correct})

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            # if (class_acc >= best_class_acc):
            #     best_class_acc = class_acc
            # log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            # log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            # Save model every 20 epochs
            if epoch % 20 == 0:
                logger.info('Saving periodic checkpoint...')
                savepath = str(checkpoints_dir) + f'/model_epoch_{epoch}.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'instance_acc': instance_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)


            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    # 'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
