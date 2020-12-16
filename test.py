import argparse
import sys
import os
import numpy as np
import cv2
import scipy.io
import copy

import torch
import torch.utils.data

from config import LFW_DATA_DIR, CFP_DATA_DIR
from models.model import Model
from dataloader.LFW_loader import LFW
from dataloader.CFP_loader import CFP
from utils import evaluation_10_fold, parseList

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(testloader, net, save_path, folds, flags):
    net.eval()
    featureLs = None
    featureRs = None
    count = 0
    for data in testloader:
        for i in range(len(data)):
            data[i] = data[i].to(device)
        count += data[0].size(0)
        print('extracting deep features from the face pair {}...'.format(count))
        res = [net(d).data.cpu().numpy()for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        break

    result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
    scipy.io.savemat(save_path, result)
    
def main(args):

    if args.target_dataset == 'lfw':
        nl, nr, flods, flags = parseList(args.dataset_path, args.target_dataset)
        dataset = LFW(nl, nr)
    elif args.target_dataset == 'cfp':
        nl, nr, flods, flags = parseList(args.dataset_path, args.target_dataset)
        dataset = CFP(nl, nr)
        
    test_loader = torch.utils.data.DataLoader(lfw_dataset,
                                              batch_size=32,
                                              shuffle=False,
                                              num_workers=0,
                                              drop_last=False)
    net = Model(args.model_name)

    ckpt = torch.load(args.checkpoint_path)
    net.load_state_dict(ckpt['net_state_dict']) # ['net_state_dict']

    result_path = args.target_dataset+'_test_result.mat'   
    test(test_loader, net, result_path, flods, flags)
    ACCs = evaluation_10_fold(result_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test lighweight face verification model')
    parser.add_argument('--target_dataset', type=str, default='lfw', help='The name of target dataset ')
    parser.add_argument('--dataset_path', type=str, default='', help='The path of target dataset ')
    parser.add_argument('--model_name', type=str, default='mobileface_csp', help='The name of target model ')
    parser.add_argument('--checkpoint_path', type=str, default='', help='The checkpoint path of the model ')
    parser.add_argument('--feature_save_dir', type=str, default='',
                        help='The path of the extract features save, must be .mat file')
    
    args = parser.parse_args()
    main(args)
