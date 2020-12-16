import argparse
import os
import time
import numpy as np
import scipy.io
import tqdm
from datetime import datetime

import torch.utils.data
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision import datasets, transforms

from models.model import Model
from models.model import ArcMarginProduct
from dataloader.CASIA_Face_loader import CASIA_Face
from dataloader.LFW_loader import LFW
from test import test
from utils import init_log, print_model_info, evaluation_10_fold, parseList

from config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU
from config import CASIA_DATA_DIR, LFW_DATA_DIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # initialize
    start_epoch = 1
    save_dir = os.path.join(SAVE_DIR, MODEL_PRE + 'v2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # define transform and dataset
    data_transform = transforms.Compose([
        transforms.Resize((112,96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    trainset = datasets.ImageFolder(root=CASIA_DATA_DIR, 
                                    transform=data_transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=BATCH_SIZE[0],
                                              shuffle=True,
                                              num_workers=0,
                                              drop_last=False)

    # nl: left_image_path
    # nr: right_image_path
    # flags: same or diff
    nl, nr, folds, flags = parseList(root=LFW_DATA_DIR, target_data='lfw')
    testdataset = LFW(nl, nr)
    testloader = torch.utils.data.DataLoader(testdataset,
                                             batch_size=BATCH_SIZE[1],
                                             shuffle=False, 
                                             num_workers=0,
                                             drop_last=False)

    # define model
    net = Model(args.model_name)
    print_model_info(net, (3, 112, 96))
    # 128 -> model last channel 
    ArcMargin = ArcMarginProduct(128, len(trainset.classes), s=32, m=0.5)

    if RESUME:
        ckpt = torch.load(RESUME)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # define optimizers
    ignored_params = list(map(id, net.model.linear1.parameters()))
    ignored_params += list(map(id, ArcMargin.weight))
    prelu_params_id = []
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer_ft = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.model.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': ArcMargin.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=0.1, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)
    
    # gpu setting 
    net = net.to(device)
    ArcMargin = ArcMargin.to(device)
    criterion = torch.nn.CrossEntropyLoss()


    best_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, TOTAL_EPOCH+1):
        exp_lr_scheduler.step()   
        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
        net.train()

        train_total_loss = 0.0
        total = 0
        since = time.time()
        for _, data in enumerate(tqdm.tqdm(trainloader)):
            img, label = data[0].to(device), data[1].to(device)
            batch_size = img.size(0)
            optimizer_ft.zero_grad()

            raw_logits = net(img)

            output = ArcMargin(raw_logits, label)
            total_loss = criterion(output, label)
            total_loss.backward()
            optimizer_ft.step()

            train_total_loss += total_loss.item() * batch_size
            total += batch_size
            break

        exp_lr_scheduler.step()
        train_total_loss = train_total_loss / total
        time_elapsed = time.time() - since
        loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s'\
            .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
        _print(loss_msg)

        # test model on lfw
        if epoch % TEST_FREQ == 0:
            test(testloader, net, 'tmp_result.mat', folds, flags)
            accs = evaluation_10_fold('tmp_result.mat')
            _print('    ave: {:.4f}'.format(np.mean(accs) * 100))
    
        # save model
        if epoch % SAVE_FREQ == 0:
            msg = 'Saving checkpoint: {}'.format(epoch)
            _print(msg)
            net_state_dict = net.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            torch.save({
                'epoch': epoch,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train lighweight face verification model')
    parser.add_argument('--model_name', type=str, default='mobileface_csp', help='model name (mobileface_csp, mobileface)')
    parser.add_argument('--batch_size', type=int, default='32', help='model name (mobileface_csp, mobileface)')

    args = parser.parse_args()

    main(args)
