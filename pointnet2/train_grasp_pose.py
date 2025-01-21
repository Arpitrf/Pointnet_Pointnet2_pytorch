"""
Script to train a model for screw axis prediction on the bimanual dataset.
Run as:
    python train_bimanual_axis.py --obj tissue --model pointnet_reg --normal --log_dir pointnet_reg --gpu 0 --epoch 1001
"""
import argparse
import os
import os.path as osp
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_utils.graspDataLoader import GraspDataset
# from visualizer.bimanual_utils import visualize_pcl_axis
# from visualizer.tensorboard_utils import log_val_q
from open3d.visualization.tensorboard_plugin import summary
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--data_dir', type=str, default='data/bimanual', help='data directory')
    parser.add_argument('--obj', type=str, default='tissue', help='object name')
    parser.add_argument('--task', type=str, default='axis', help='Choose from: contact, axis')
    parser.add_argument('--model', type=str, default='text_pointnet2_reg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=501, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--mat_diff_loss_scale', type=float, default=0.001, help='weight for matching different loss')
    parser.add_argument('--axis_loss_scale', type=float, default=1.0, help='weight for axis loss')

    return parser.parse_args()


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
    exp_dir = osp.join(exp_dir, 'axis_reg', args.obj)
    os.makedirs(exp_dir, exist_ok=True)
    exp_dir = Path(exp_dir)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    viz_dir = exp_dir.joinpath('viz/')
    viz_dir.mkdir(exist_ok=True)

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

    datapath = osp.join(args.data_dir, args.obj)

    TRAIN_DATASET = GraspDataset(split="train")
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    VAL_DATASET = GraspDataset(split="val")
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of val data is: %d" % len(VAL_DATASET))

    writer = SummaryWriter(f'runs/{args.log_dir}')

    # # testing dataloader
    # next(iter(trainDataLoader))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model().cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_total_loss = np.inf

    for epoch in range(start_epoch, args.epoch):
        train_metrics = {}
        train_losses = {'total': []}

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, target, text_description) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            print("IN TRAIN LOOP points, grasp, text_description: ", points.shape, target.shape, len(text_description))

            # breakpoint()
            points = points.data.numpy()
            # TODO: verify augmentations. Change later
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.float().cuda()
            points = points.transpose(2, 1)

            text_description = list(text_description)

            grasp_pred, _ = classifier(points, text_description)
            
            # # Visualize the 0th element in the batch for pred and targets for training
            # vis_points = points[0].transpose(1, 0).cpu().detach().numpy()
            # vis_axis_pred = axis_pred.cpu().detach().numpy()
            # vis_target = target.cpu().detach().numpy()
            # if args.use_q:
            #     vis_axis_pred = vis_axis_pred[0].reshape(2,3)
            #     vis_target = vis_target[0].reshape(2,3)
            # else:
            #     vis_axis_pred = vis_axis_pred[0]
            #     vis_target = vis_target[0]
            # visualize_pcl_axis([vis_axis_pred, vis_target], vis_points.shape[0], vis_points[:, :3], savepath='/home/arpit/test_projects/bimanual_predictor/temp.png', use_q=args.use_q)

            # # remove later
            # with torch.no_grad():
            #     for i, a in enumerate(axis_pred):
            #         axis_pred[i] = axis_pred[i] / torch.norm(axis_pred[i])
            #     print('axis_pred: ', axis_pred.shape, type(axis_pred), axis_pred.dtype)
            # print('target: ', target.shape, type(target), target.dtype)
            loss_dict = criterion(grasp_pred, target)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            train_losses['total'].append(loss.item())

        train_metrics['total_loss'] = np.mean(train_losses['total'])
        log_string('Train Loss:\t Total: {:.4f}'.format(
            train_metrics['total_loss']))
        
        for k, v in train_metrics.items():
            writer.add_scalar(
                f'train_loss/{k}', v, global_step=epoch)

        with torch.no_grad():
            val_metrics = {}
            val_losses = {'total': []}

            classifier = classifier.eval()

            for batch_id, (points, target, text_description) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, target = points.float().cuda(), target.float().cuda()
                points = points.transpose(2, 1)
                text_description = list(text_description)
    
                grasp_pred, _ = classifier(points, text_description)
                # remove later
                # for i, a in enumerate(axis_pred):
                #     axis_pred[i] = axis_pred[i] / torch.norm(axis_pred[i])
                #     print("val: axis_pred, target: ", axis_pred[i], target[i])
                loss_dict = criterion(grasp_pred, target)
                val_losses['total'].append(loss_dict['total'].item())
                print("val_losses['total']: ", val_losses['total'])

            # save visualization of predictions
            tensorboard_log = {
                'target_q': [],
                'pred_q': [],
                'q_diff': []
            }
            # if epoch % 100 == 0:
            #     points = points.transpose(2, 1).cpu().numpy()
            #     axis_pred = axis_pred.cpu().data.numpy()
            #     target = target.cpu().data.numpy()
            #     savedir = viz_dir.joinpath(f'{epoch:03d}')
            #     savedir.mkdir(exist_ok=True)
            #     for i in range(cur_batch_size):
            #         axisp = axis_pred[i, :]
            #         axisl = target[i, :]
            #         tensorboard_log['target_q'].append([round(e, 2) for e in axisl])
            #         tensorboard_log['pred_q'].append([round(e, 2) for e in axisp])
            #         tensorboard_log['q_diff'].append(np.linalg.norm(axisp - axisl))
            #         savepath = savedir.joinpath(f'{i:02d}.png')
            #         if args.use_q and args.use_s:
            #             axisl = axisl.reshape(2,3)
            #             axisp = axisp.reshape(2,3)
            #         visualize_pcl_axis([axisl, axisp], NUM_POINT, points[i,:,:3], savepath, use_q=args.use_q, use_s=args.use_s, writer=writer, epoch=epoch)
            #     print("tensorboard_log: ", tensorboard_log.keys(), len(tensorboard_log['pred_q']))  
            #     log_val_q(tensorboard_log, writer, epoch)
            
            val_metrics['total_loss'] = np.mean(val_losses['total'])

        log_string('Epoch {} Val loss\t total: {:.4f}' .format(
            epoch + 1, val_metrics['total_loss']))
        
        for k, v in val_metrics.items():
            writer.add_scalar(
                f'val_loss/{k}', v, global_step=epoch)

        
        if epoch % 500 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + f'/{epoch}.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_total_loss': train_metrics['total_loss'],
                'val_total_loss': val_metrics['total_loss'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')
        
        if (val_metrics['total_loss'] <= best_total_loss):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_total_loss': train_metrics['total_loss'],
                'val_total_loss': val_metrics['total_loss'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        # remove later
        # if epoch > 3:
        if val_metrics['total_loss'] < best_total_loss:
            best_total_loss = val_metrics['total_loss']
        log_string('Best total loss is: %.5f' % best_total_loss)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)