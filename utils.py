from __future__ import print_function
import os
import logging
import numpy as np
import scipy
from torch import nn
import torch 

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def print_model_info(model, input_shape):
    from thop import profile
    try:
        from torchsummaryX import summary
        summary(model, torch.randn((1, ) + input_shape))
    except:
        from torchsummary import summary
        summary(model, input_shape, device='cpu')

    input_shape = (1, ) + input_shape
    input_sample = torch.randn(input_shape)
    total_ops, total_params = profile(model, (input_sample, ), verbose=False)
    print("%s | %s | %s" % ("Params(M)", "FLOPs(G)", "FLOPs(M)"))
    print("%.2fM | %.2fG | %.2fM" % (total_params / (1000**2), total_ops / (1000**3), total_ops / (1000**2)))

def parseList(root, target_data):
    if target_data == 'lfw':
        txt_path = os.path.join(root, 'lfw_test_pair.txt')
        folder_name = 'lfw-align-128'
    elif target_data == 'cfp':
        txt_path = os.path.join(root, 'cfp-ff-pair.txt')
        folder_name = 'aligned'

    with open(txt_path) as f:
        pairs = f.read().splitlines()#[1:]
    
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split(' ')
        if int(p[2]) == 1:
            nameL = os.path.join(root, folder_name, p[0])
            nameR = os.path.join(root, folder_name, p[1])
            fold = i // 600
            flag = 1
        
        else:
            nameL = os.path.join(root, folder_name, p[0])
            nameR = os.path.join(root, folder_name, p[1])
            fold = i // 600
            flag = -1

        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)
    return [nameLs, nameRs, folds, flags]



def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def evaluation_10_fold(root):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)

    return ACCs

if __name__ == '__main__':
    pass
