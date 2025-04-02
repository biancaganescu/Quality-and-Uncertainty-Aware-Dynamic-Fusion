import sys 
import os 
sys.path.append(os.getcwd())
sys.path.append('/home/bianca/Code/MultiBench/mm_health_bench/mmhb')
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from unimodals.common_models import VGG11Slim, MLP, ReportTransformer
from datasets.imdb.get_data import get_dataloader
from training_structures.unimodal import train, test
from torch.utils.data import DataLoader, TensorDataset
# from mm_health_bench.mmhb.loader import *
# from mm_health_bench.mmhb.utils import Config
from chestx_utils import *
from torchvision import models
def compute_pos_weights(train_loader):
    """
    Compute positive weights for BCEWithLogitsLoss based on class frequencies
    
    Returns:
        torch.Tensor: Tensor of shape [num_classes] with positive weights
    """
    num_pos = torch.zeros(14)  # Assuming 14 classes for chest X-ray
    num_neg = torch.zeros(14)
    
    for _, _, targets in train_loader:
        num_pos += targets.sum(dim=0)
        num_neg += (1 - targets).sum(dim=0)
    
    # Compute ratio of negative to positive examples
    pos_weights = num_neg / (num_pos + 1e-5)  # Add small epsilon to prevent division by zero
    
    # Cap weights to prevent extreme values (optional)
    pos_weights = torch.clamp(pos_weights, min=0.5, max=10.0)
    
    return pos_weights.cuda()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("chestx",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--mod", type=int, default=0, help="0: text; 1: image")
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--measure", action='store_true', help='no training')
    argparser.add_argument("--dir", default='chestx/', help='folder to store results')
    args = argparser.parse_args()

    
    if not os.path.exists("./log/" + args.dir):
        os.makedirs("./log/" + args.dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    modality = 'image' if args.mod == 1 else 'text'
    model_file = "./log/" + args.dir + "model_" + modality + ".pt"
    head_file = "./log/" + args.dir + "head_" + modality + ".pt"

    log1, log2 = [], []
    for n in range(args.n_runs):
        if args.mod == 0:
            # model = MLP(256, 256, 256).cuda()
            model = ReportTransformer(30522, 256, 256).cuda()
            head = MLP(256, 256, 14).cuda() 
        else:
            model = VGG11Slim(256).cuda()
            head = MLP(256, 256, 14).cuda() 

        train_data, val_data, test_data = get_data(batch_size=64, num_workers=4)

        if not args.eval_only:
            train(model, head, train_data, val_data, 10, early_stop=True, task="multilabel",
                    save_encoder=model_file, save_head=head_file,
                    modalnum=args.mod, optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=0.01,
                    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=compute_pos_weights(train_data)))

        print(f"Testing model {model_file} and {head_file}:")
        model = torch.load(model_file).cuda()
        head = torch.load(head_file).cuda()
        model.eval()
        tmp = test(model, head, test_data, "default", modality, criterion=torch.nn.BCEWithLogitsLoss(pos_weight=compute_pos_weights(train_data)), task="multilabel", modalnum=args.mod, no_robust=True)
        log1.append(tmp['f1_micro'])
        log2.append(tmp['f1_macro'])

    print(log1, log2)
    print(f'Finish {args.n_runs} runs')
    print(f'f1 micro {np.mean(log1) * 100:.2f} ± {np.std(log1) * 100:.2f}')
    print(f'f1 macro {np.mean(log2) * 100:.2f} ± {np.std(log2) * 100:.2f}')
    
    
        
