import numpy as np
import scipy.misc
import imageio
# pip install scipy==1.1.0

import torch
class LFW(object):
    def __init__(self, imgl, imgr):

        self.imgl_list = imgl
        self.imgr_list = imgr

    def __getitem__(self, index):
        imgl = imageio.imread(self.imgl_list[index]) # 128 128
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        imgr = imageio.imread(self.imgr_list[index]) # 128 128
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            #imglist[i] = scipy.misc.imresize(imglist[i], (112, 96)) # yeogi
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]
        
        return imgs

    def __len__(self):
        return len(self.imgl_list)


if __name__ == '__main__':
    pass

    # lfw_dir = 'C:\\Users\\pc\\Desktop\\datasets\\lfw-align-128'

    # with open(os.path.join(root, 'lfw_test_pair.txt')) as f:
    #     pairs = f.read().splitlines()#[1:]
    # folder_name = 'lfw-align-128'
    # nameLs = []
    # nameRs = []
    # folds = []
    # flags = []
    # for i, p in enumerate(pairs):
    #     p = p.split(' ')
    #     if int(p[2]) == 1:
    #         nameL = os.path.join(root, folder_name, p[0])
    #         nameR = os.path.join(root, folder_name, p[1])
    #         fold = i // 600
    #         flag = 1
        
    #     else:
    #         nameL = os.path.join(root, folder_name, p[0])
    #         nameR = os.path.join(root, folder_name, p[1])
    #         fold = i // 600
    #         flag = -1

    #     nameLs.append(nameL)
    #     nameRs.append(nameR)
    #     folds.append(fold)
    #     flags.append(flag)
    # return [nameLs, nameRs, folds, flags]