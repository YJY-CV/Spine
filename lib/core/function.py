# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.evaluate import Angles_accuracy
from core.evaluate import Points_accuracy
from core.evaluate import CMAE_accuracy
from core.inference import get_final_preds
from core.inference import get_final_points
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, pointcri, anglecri, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossPoint = AverageMeter()
    lossScore = AverageMeter()
    accPearson = AverageMeter()
    accMAE = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta, points) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        points = points.cuda(non_blocking=True)
        input_w = config.MODEL.IMAGE_SIZE[0]
        input_h = config.MODEL.IMAGE_SIZE[1]
        #import pdb
        #pdb.set_trace()
        if isinstance(outputs, list):
            output = outputs
            scoreloss = criterion(output, target, target_weight)
            pointloss=pointcri(output, points, input_w, input_h)
            """
            if epoch<51:
                aa = 0.999
            else:
                aa = 0.9
            """
            
            aa = 1
            loss = (1-aa)*pointloss+aa*scoreloss
        else:
            output = outputs
            scoreloss = criterion(output, target, target_weight)
            loss = scoreloss

        # loss = criterion(output, target, target_weight)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        #lossPoint.update(pointloss.item(), input.size(0))
        lossScore.update(scoreloss.item(), input.size(0))
        #import pdb
        #pdb.set_trace()
        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)
        
        #_, avg_pearson_point, _1, avg_MAE_point, cnt_point, pred_points = Points_accuracy(output[1].detach().cpu().numpy(), points.detach().cpu().numpy())
        #accPearson.update(avg_pearson_point, cnt_point)
        #accMAE.update(avg_MAE_point, cnt_point)
        
        # import pdb
        # pdb.set_trace()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'lossScore {scoreloss.val:.5f} ({scoreloss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, scoreloss=lossScore, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output, prefix)


def validate(config, val_loader, val_dataset, model, criterion, pointcri, anglecri, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    #lossAngle = AverageMeter()
    lossPoint = AverageMeter()
    lossScore = AverageMeter()
    accPearson = AverageMeter()
    accMAE = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    #all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 2), dtype=np.float32)        #ori landmark model
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_preds_point = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    #all_boxes = np.zeros((num_samples, 6))
    all_boxes = np.zeros((num_samples, 22))
    all_boxes_point = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta, points) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            
            if isinstance(outputs, list):
                output = outputs
                # output = outputs[-1]
            else:
                output = outputs
            # output = output[0]
            #import pdb
            #pdb.set_trace()
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[0]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            points = points.cuda(non_blocking=True)
            input_w = config.MODEL.IMAGE_SIZE[0]
            input_h = config.MODEL.IMAGE_SIZE[1]
            scoreloss = criterion(output, target, target_weight)
            #pointloss = pointcri(output, points, input_w, input_h)
            aa = 1
            #loss = 1*angleloss + 0.1*pointloss + 0*scoreloss
            #loss = (1-aa)*pointloss + aa*scoreloss
            loss = scoreloss
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            #lossPoint.update(pointloss.item(), input.size(0))
            lossScore.update(scoreloss.item(), input.size(0))
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)
            
            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            r = meta['rotation'].numpy()
            score = meta['score'].numpy()
            w_rate=meta['w_rate']
            h_rate=meta['h_rate']
            box_list = meta['box_list'].numpy()
            id = meta['id'].numpy()
            joints_vis = meta['joints_vis'][:, :, 0].numpy()   #shape = [num_joints]
            
            scoremap_height = output.shape[2]
            scoremap_width = output.shape[3]
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            
            all_boxes[idx:idx + num_images, 6:9] = box_list[:, 0:3]
            all_boxes[idx:idx + num_images, 9] = id
            all_boxes[idx:idx + num_images, 10:22] = joints_vis[:, 0:12]
            
            image_path.extend(meta['image'])
            #import pdb
            #pdb.set_trace()
            idx += num_images
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'lossScore {scoreloss.val:.5f} ({scoreloss.avg:.5f})\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                    scoreloss=lossScore, loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output, prefix)
        #import pdb
        #pdb.set_trace()
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )
        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            
            
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            
            writer_dict['valid_global_steps'] = global_steps + 1

    
    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    #if len(full_arch_name) > 15:
        #full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
