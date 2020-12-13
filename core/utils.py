from __future__ import print_function
import os
import logging
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


if __name__ == '__main__':
    pass
