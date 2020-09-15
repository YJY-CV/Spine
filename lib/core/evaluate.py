# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    # import pdb
    # pdb.set_trace()
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def Angles_accuracy(output, angles, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    # import pdb
    # pdb.set_trace()

    # angles_size = output.size(0)
    # angles_num = output.size(1)
    # angles_pre = output.squeeze()
    # angles = angles.squeeze()
    # angle_loss = 0.0
    acc = np.zeros((len(output) + 1))
    avg_acc = 0
    cnt = 0
    angles_idx=0
    dists=np.zeros((len(output),1))
    for angles_pre,angleslab in zip(output,angles):
        preatan = np.arctan(np.sin(angles_pre)/np.cos(angles_pre))
        labatan = np.arctan(np.sin(angleslab)/ np.cos(angleslab))
        acc[angles_idx] = (1-(np.mean(preatan*labatan)-np.mean(preatan)*np.mean(labatan))/(np.std(preatan)*np.std(labatan)))

        # import pdb
        # pdb.set_trace()


    return acc, np.mean(acc), cnt, output

def Points_accuracy(output, points, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    #import pdb
    #pdb.set_trace()
    num_joints = int(output.shape[1]/2)

    # angles_size = output.size(0)
    # angles_num = output.size(1)
    # points_pred = output.squeeze()
    # angles = angles.squeeze()
    # angle_loss = 0.0
    acc_pearson = np.zeros((len(output)))
    acc_MAE = np.zeros((len(output)))
    cnt = 0
    pearson_idx = 0
    MAE_idx = 0
    dists = np.zeros((len(output),1))
    
    # Pearson
    for points_pred, points_gt in zip(output,points):
        acc_pearson[pearson_idx] = (np.mean(points_pred*points_gt)-np.mean(points_pred)*np.mean(points_gt))/(np.std(points_pred)*np.std(points_gt))+1
        pearson_idx += 1
    
    # MAE
    for points_pred, points_gt in zip(output,points):
        acc_MAE[MAE_idx] = np.mean(abs(points_pred-points_gt))
        MAE_idx += 1
    
    point=np.zeros((len(output),num_joints,2))
    for idx in range(len(output)):
        for point_idx in range(num_joints):
            point[idx][point_idx][0]=output[idx][point_idx]
            point[idx][point_idx][1]=output[idx][point_idx+num_joints]

    return acc_pearson, np.mean(acc_pearson), acc_MAE, np.mean(acc_MAE), pearson_idx, point


def calc_mid_vec(x0, y0, x1, y1, x2, y2, x3, y3):
    x_m = float(x1+x3-x2-x0)/2.0
    y_m = float(y1+y3-y2-y0)/2.0
    return x_m, y_m

def calc_cos(x0, y0, x1, y1):
    if x0==x1 and y0==y1:
        cos_m = 1.0
    else:
        innerP_m = x0*x1+y0*y1
        mod_1 = np.sqrt(x0**2+y0**2)
        mod_2 = np.sqrt(x1**2+y1**2)
        cos_m = innerP_m/(mod_1*mod_2)
    return cos_m

def find_min_cos(vec_m):
    cos_min = 1.0
    p1 = 0
    p2 = 16
    for i in range(16):
        for j in range(i+1, 17):
            cos_m = calc_cos(vec_m[i][0], vec_m[i][1], vec_m[j][0], vec_m[j][1])
            if cos_m<cos_min:
                p1 = i
                p2 = j
                cos_min = cos_m
    return p1, p2, cos_min

def find_min_cos_up(vec_m, p2):
    cos_min = 1.0
    p1 = 0
    for i in range(0, p2):
        cos_m = calc_cos(vec_m[i][0], vec_m[i][1], vec_m[p2][0], vec_m[p2][1])
        if cos_m<cos_min:
            p1 = i
            cos_min = cos_m
    return p1, cos_min

def find_min_cos_down(vec_m, p1):
    cos_min = 1.0
    p2 = 16
    for i in range(p1, 17):
        cos_m = calc_cos(vec_m[p1][0], vec_m[p1][1], vec_m[i][0], vec_m[i][1])
        if cos_m<cos_min:
            p2 = i
            cos_min = cos_m
    return p2, cos_min

def isS(mid_p_v_gt):
    num = len(mid_p_v_gt)
    mat = np.zeros(num-2)
    if mid_p_v_gt[0][1]-mid_p_v_gt[num-1][1]==0:
        delta_y = 0.000001
    else:
        delta_y = mid_p_v_gt[0][1]-mid_p_v_gt[num-1][1]
    if mid_p_v_gt[0][0]-mid_p_v_gt[num-1][0]==0:
        delta_x = 0.000001
    else:
        delta_x = mid_p_v_gt[0][0]-mid_p_v_gt[num-1][0]
    for i in range(num-2):
        mat[i] = (mid_p_v_gt[i][1]-mid_p_v_gt[num-1][1])/delta_y - (mid_p_v_gt[i][0]-mid_p_v_gt[num-1][0])/delta_x
    if np.sum(np.sum(mat.reshape([num-2,1])*mat)) != np.sum(np.sum(np.abs(mat.reshape([num-2,1])*mat))):
        ff = True
    else:
        ff = False
    return ff

def CMEAN(A1, A2, A3):
    #import pdb
    #pdb.set_trace()
    A1=A1/180*np.pi
    A2=A2/180*np.pi
    A3=A3/180*np.pi
    xmean = (np.cos(A1)+np.cos(A2)+np.cos(A3))/3.
    ymean = (np.sin(A1)+np.sin(A2)+np.sin(A3))/3.
    cmean = np.arctan(ymean/xmean)*180/np.pi
    return cmean


def CMAE_accuracy(output, points, h, hm_type='gaussian', thr=0.5):

    acc_CMAE = np.zeros((len(output)))
    acc_SMAPE = np.zeros((len(output)))
    CMAE_idx = 0
    
    # MAE
    for points_pred, points_gt in zip(output,points):
        
        x = np.zeros((68))
        y = np.zeros((68))
        x_gt = np.zeros((68))
        y_gt = np.zeros((68))
        for point_idx in range(68):
            #import pdb
            #pdb.set_trace()
            x[point_idx]=points_pred[point_idx]
            y[point_idx]=points_pred[point_idx+68]
            x_gt[point_idx]=points_gt[0][point_idx]
            y_gt[point_idx]=points_gt[0][point_idx+68]
        
        #for prediction Cobb
        vec_m = []
        mid_p_v = []
        for i in range(17):
            x_m, y_m = calc_mid_vec(x[4*i], y[4*i], x[4*i+1], y[4*i+1], x[4*i+2], y[4*i+2], x[4*i+3], y[4*i+3])
            vec_m.append([x_m, y_m])
            mid_p_v.append([(x[4*i]+x[4*i+1])/2., (y[4*i]+y[4*i+1])/2.])
            mid_p_v.append([(x[4*i+2]+x[4*i+3])/2., (y[4*i+2]+y[4*i+3])/2.])
            
        pos1, pos2, cos_min = find_min_cos(vec_m)
        
        if not isS(mid_p_v):
            pos0 = 0
            pos3 = 16
            cos_min_up = calc_cos(vec_m[0][0], vec_m[0][1], vec_m[pos1][0], vec_m[pos1][1])
            cos_min_down = calc_cos(vec_m[pos2][0], vec_m[pos2][1], vec_m[16][0], vec_m[16][1])
        else:
            if (mid_p_v[2*pos1+1][1]+mid_p_v[2*pos2+1][1])<h:
                pos0, cos_min_up = find_min_cos_up(vec_m, pos1)
                pos3, cos_min_down = find_min_cos_down(vec_m, pos2)
            else:
                pos0, cos_min_up = find_min_cos_up(vec_m, pos1)
                pos3, cos_min_down = find_min_cos_up(vec_m, pos0)
        
        
        cobb = np.arccos(cos_min)*180/np.pi
        cobb_up = np.arccos(cos_min_up)*180/np.pi
        cobb_down = np.arccos(cos_min_down)*180/np.pi
        
        #for GT Cobb
        vec_m_gt = []
        mid_p_v_gt = []
        for i in range(17):
            x_m_gt, y_m_gt = calc_mid_vec(x_gt[4*i], y_gt[4*i], x_gt[4*i+1], y_gt[4*i+1], x_gt[4*i+2], y_gt[4*i+2], x_gt[4*i+3], y_gt[4*i+3])
            vec_m_gt.append([x_m_gt, y_m_gt])
            mid_p_v_gt.append([(x_gt[4*i]+x_gt[4*i+1])/2., (y_gt[4*i]+y_gt[4*i+1])/2.])
            mid_p_v_gt.append([(x_gt[4*i+2]+x_gt[4*i+3])/2., (y_gt[4*i+2]+y_gt[4*i+3])/2.])
            
        pos1_gt, pos2_gt, cos_min_gt = find_min_cos(vec_m_gt)
        
        if not isS(mid_p_v_gt):
            pos0_gt = 0
            pos3_gt = 16
            cos_min_up_gt = calc_cos(vec_m_gt[0][0], vec_m_gt[0][1], vec_m_gt[pos1_gt][0], vec_m_gt[pos1_gt][1])
            cos_min_down_gt = calc_cos(vec_m_gt[pos2_gt][0], vec_m_gt[pos2_gt][1], vec_m_gt[16][0], vec_m_gt[16][1])
        else:
            if (mid_p_v_gt[2*pos1_gt+1][1]+mid_p_v_gt[2*pos2_gt+1][1])<h:
                pos0_gt, cos_min_up_gt = find_min_cos_up(vec_m_gt, pos1_gt)
                pos3_gt, cos_min_down_gt = find_min_cos_down(vec_m_gt, pos2_gt)
            else:
                pos0_gt, cos_min_up_gt = find_min_cos_up(vec_m_gt, pos1_gt)
                pos3_gt, cos_min_down_gt = find_min_cos_up(vec_m_gt, pos0_gt)
        
        cobb_gt = np.arccos(cos_min_gt)*180/np.pi
        cobb_up_gt = np.arccos(cos_min_up_gt)*180/np.pi
        cobb_down_gt = np.arccos(cos_min_down_gt)*180/np.pi
        delta_u = abs(cobb_up_gt-cobb_up)
        delta_m = abs(cobb_gt-cobb)
        delta_d = abs(cobb_down_gt-cobb_down)
        acc_CMAE[CMAE_idx] = CMEAN(delta_u, delta_m, delta_d)
        acc_SMAPE[CMAE_idx] = (delta_m + delta_u + delta_d)/(cobb_up_gt+cobb_up + cobb_gt-cobb+cobb+ cobb_down_gt+cobb_down)*100
        CMAE_idx += 1
        #if acc_CMAE[CMAE_idx-1]<0:
        #    import pdb
        #    pdb.set_trace()
    

    return acc_CMAE, np.mean(acc_CMAE), acc_SMAPE, np.mean(acc_SMAPE), CMAE_idx

def CMAE_accuracy_for_GT_transform(points_gt, points_trans, h_gt, h_trans):

    acc_CMAE = 0
    way_trans = 0
    way_gt = 0
    
    x = np.zeros((68),dtype='float32')
    y = np.zeros((68),dtype='float32')
    x_gt = np.zeros((68),dtype='float32')
    y_gt = np.zeros((68),dtype='float32')
    for point_idx in range(68):
        #import pdb
        #pdb.set_trace()
        x[point_idx]=points_trans[point_idx][0]
        y[point_idx]=points_trans[point_idx][1]
        x_gt[point_idx]=points_gt[point_idx][0]
        y_gt[point_idx]=points_gt[point_idx][1]
    
    #for trans Cobb
    vec_m = []
    mid_p_v = []
    for i in range(17):
        x_m, y_m = calc_mid_vec(x[4*i], y[4*i], x[4*i+1], y[4*i+1], x[4*i+2], y[4*i+2], x[4*i+3], y[4*i+3])
        vec_m.append([x_m, y_m])
        mid_p_v.append([(x[4*i]+x[4*i+1])/2., (y[4*i]+y[4*i+1])/2.])
        mid_p_v.append([(x[4*i+2]+x[4*i+3])/2., (y[4*i+2]+y[4*i+3])/2.])
        
    pos1, pos2, cos_min = find_min_cos(vec_m)
    
    if not isS(mid_p_v):
        pos0 = 0
        pos3 = 16
        way_trans = 1
        cos_min_up = calc_cos(vec_m[0][0], vec_m[0][1], vec_m[pos1][0], vec_m[pos1][1])
        cos_min_down = calc_cos(vec_m[pos2][0], vec_m[pos2][1], vec_m[16][0], vec_m[16][1])
    else:
        if (mid_p_v[2*pos1+1][1]+mid_p_v[2*pos2+1][1])<h_trans:
            way_trans = 2
            pos0, cos_min_up = find_min_cos_up(vec_m, pos1)
            pos3, cos_min_down = find_min_cos_down(vec_m, pos2)
        else:
            way_trans = 3
            pos0, cos_min_up = find_min_cos_up(vec_m, pos1)
            pos3, cos_min_down = find_min_cos_up(vec_m, pos0)
    
    
    cobb = np.arccos(cos_min)*180/np.pi
    cobb_up = np.arccos(cos_min_up)*180/np.pi
    cobb_down = np.arccos(cos_min_down)*180/np.pi
    
    #for GT Cobb
    vec_m_gt = []
    mid_p_v_gt = []
    for i in range(17):
        x_m_gt, y_m_gt = calc_mid_vec(x_gt[4*i], y_gt[4*i], x_gt[4*i+1], y_gt[4*i+1], x_gt[4*i+2], y_gt[4*i+2], x_gt[4*i+3], y_gt[4*i+3])
        vec_m_gt.append([x_m_gt, y_m_gt])
        mid_p_v_gt.append([(x_gt[4*i]+x_gt[4*i+1])/2., (y_gt[4*i]+y_gt[4*i+1])/2.])
        mid_p_v_gt.append([(x_gt[4*i+2]+x_gt[4*i+3])/2., (y_gt[4*i+2]+y_gt[4*i+3])/2.])
        
    pos1_gt, pos2_gt, cos_min_gt = find_min_cos(vec_m_gt)
    
    if not isS(mid_p_v_gt):
        pos0_gt = 0
        pos3_gt = 16
        way_gt = 1
        cos_min_up_gt = calc_cos(vec_m_gt[0][0], vec_m_gt[0][1], vec_m_gt[pos1_gt][0], vec_m_gt[pos1_gt][1])
        cos_min_down_gt = calc_cos(vec_m_gt[pos2_gt][0], vec_m_gt[pos2_gt][1], vec_m_gt[16][0], vec_m_gt[16][1])
    else:
        if (mid_p_v_gt[2*pos1_gt+1][1]+mid_p_v_gt[2*pos2_gt+1][1])<h_gt:
            way_gt = 2
            pos0_gt, cos_min_up_gt = find_min_cos_up(vec_m_gt, pos1_gt)
            pos3_gt, cos_min_down_gt = find_min_cos_down(vec_m_gt, pos2_gt)
        else:
            way_gt = 3
            pos0_gt, cos_min_up_gt = find_min_cos_up(vec_m_gt, pos1_gt)
            pos3_gt, cos_min_down_gt = find_min_cos_up(vec_m_gt, pos0_gt)
    
    cobb_gt = np.arccos(cos_min_gt)*180/np.pi
    cobb_up_gt = np.arccos(cos_min_up_gt)*180/np.pi
    cobb_down_gt = np.arccos(cos_min_down_gt)*180/np.pi
    
    delta_up = abs(round(cobb_up_gt,4)-round(cobb_up,4))
    delta_middle =abs(round(cobb_gt,4)-round(cobb,4))
    delta_down = abs(round(cobb_down_gt,4)-round(cobb_down,4))
    
    acc_CMAE = CMEAN(delta_up, delta_middle, delta_down)    

    return acc_CMAE, delta_up, delta_middle, delta_down, way_gt, way_trans


