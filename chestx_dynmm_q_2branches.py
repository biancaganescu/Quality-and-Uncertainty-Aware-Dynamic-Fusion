import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.getcwd())
from chestx_utils import *
from unimodals.common_models import MLP, Linear, MaxOut_MLP, ReportTransformer, Transformer, Sequential, Identity
from fusions.common_fusions import Concat
from ModalityDynMM.training_structures_dynmm.Supervised_Learning import train, test, MMDL
from noise import get_noisy_data_loaders

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class QualityAssessor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(QualityAssessor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

class DynMMNet(nn.Module):
    def __init__(self, branch_num=2, pretrain=True, freeze=True, directory = "chestx/"):
         # Add branch selection counters
        self.branch_selections = torch.zeros(branch_num)
        self.total_samples = 0
        super(DynMMNet, self).__init__()
        self.branch_num = branch_num
        self.dir = directory
        # branch 1: text network
        self.text_encoder = torch.load('./log/' + self.dir + 'model_text.pt') if pretrain else ReportTransformer(30522, 256, 256)
        self.text_head = torch.load('./log/' + self.dir + 'head_text.pt') if pretrain else MLP(256, 256, 14)

        #image network, used only for encoding in forward
        self.image_encoder = torch.load('./log/' + self.dir + 'model_image.pt') if pretrain else ReportTransformer(30522, 256, 256)
        self.image_head = torch.load('./log/' + self.dir + 'head_image.pt') if pretrain else MLP(256, 256, 14)

        # branch3: text+image late fusion
        if pretrain:
            self.branch3 = torch.load('./log/' + self.dir + 'best_lf.pt')
        else:
            encoders = [ReportTransformer(30522, 256, 256).cuda(), VGG11Slim(256).cuda()]
            head= Linear(514, 14).cuda()
            fusion = Concat()
            self.branch3 = MMDL(encoders, fusion, head)

        self.text_quality = QualityAssessor(256)
        self.image_quality = QualityAssessor(256)

        if freeze:

            self.freeze_branch(self.text_encoder)
            self.freeze_branch(self.text_head)
            self.freeze_branch(self.branch3)

        # gating network
        self.gate = MLP(514, 256, branch_num)
        self.temp = 1
        self.hard_gate = True
        self.weight_list = torch.Tensor()
        self.store_weight = False
        self.infer_mode = 0
        self.flop = torch.Tensor([2.18538, 21.82964])
    
    def reset_selection_stats(self):
        self.branch_selections = torch.zeros_like(self.branch_selections)
        self.total_samples = 0

    def freeze_branch(self, m):
        for param in m.parameters():
            param.requires_grad = False

    def reset_weight(self):
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self):
        tmp = torch.mean(self.weight_list, dim=0)
        print(f'mean branch weight {tmp[0].item():.4f}, {tmp[1].item():.4f}')
        self.store_weight = False
        return tmp[1].item()  # Return the weight for the fusion branch

    def cal_flop(self):
        tmp = torch.mean(self.weight_list, dim=0)
        total_flop = (self.flop * tmp).sum()
        print(f'Total Flops {total_flop.item():.2f}M')
        return total_flop.item()

    def forward(self, inputs):
        text_input = inputs[0]  # Shape: [batch, seq]
        image_input = inputs[1]  # Shape: [batch, h, w, c]
        
        # Get batch size
        batch_size = text_input.size(0)
        text_features = self.text_encoder(text_input)
        image_features = self.image_encoder(inputs[1]) 

        image_features = image_features.view(batch_size, -1) 
        
        text_quality = self.text_quality(text_features)
        image_quality = self.image_quality(image_features)

        text_quality_expanded = text_quality.expand_as(text_features)
        image_quality_expanded = image_quality.expand_as(image_features)
        
        # Weight features by their quality
        weighted_text_features = text_features * text_quality_expanded
        weighted_image_features = image_features * image_quality_expanded
        # Concatenate
        x = torch.cat([weighted_text_features, text_quality, weighted_image_features, image_quality], dim=1)
    
        layer_norm = nn.LayerNorm(x.size()[1:]).to(x.device)
        x = layer_norm(x)
        weight = DiffSoftmax(self.gate(x), tau=self.temp, hard=self.hard_gate)

        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))
        
        if self.hard_gate:
            # Get the index of the maximum weight for each sample in the batch
            selected_branches = torch.argmax(weight, dim=1)
            
            # Count selections for each branch
            for i in range(self.branch_num):
                self.branch_selections[i] += (selected_branches == i).sum().item()
            
            # Track total number of samples
            self.total_samples += weight.size(0)

        # Get predictions from all three branches
        pred_list = [
            self.text_head(self.text_encoder(inputs[0])),                 # Branch 1: Text only
            self.branch3(inputs)                                         # Branch 3: Late fusion
        ]
        
        if self.infer_mode > 0:
            return pred_list[self.infer_mode - 1], 0

        # Combine predictions using weights
        output = weight[:, 0:1] * pred_list[0] +  weight[:, 1:2] * pred_list[1]
                
        # Return output and average weight of fusion branch for monitoring
        return output, weight[:, 1].mean()

    def forward_separate_branch(self, inputs, path, weight_enable):  
        if weight_enable:
            x = torch.cat(inputs, dim=1)
            weight = DiffSoftmax(self.gate(x), tau=self.temp, hard=self.hard_gate)
        if path == 1:
            output = self.branch3(inputs)

        return output

    def get_selection_stats(self):
        if self.total_samples == 0:
            return "No samples processed yet."
        
        percentages = (self.branch_selections / self.total_samples) * 100
        
        stats = "Branch selection statistics:\n"
        for i in range(self.branch_num):
            stats += f"Branch {i+1}: selected {self.branch_selections[i]} times " \
                    f"({percentages[i]:.2f}% of samples)\n"
        
        return stats

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("chestx",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    argparser.add_argument("--n-runs", type=int, default=1, help="number of runs")
    argparser.add_argument("--data", type=str, default='chestx', help="dataset name")
    argparser.add_argument("--n-epochs", type=int, default=50, help="number of epochs")
    argparser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--wd", type=float, default=1e-2, help="weight decay")
    argparser.add_argument("--reg", type=float, default=0.1, help="reg loss weight")
    argparser.add_argument("--freeze", action='store_true', help='freeze branch weights')
    argparser.add_argument("--eval-only", action='store_true', help='no training')
    argparser.add_argument("--hard", action='store_true', help='hard labels')
    argparser.add_argument("--no-pretrain", action='store_true', help='train from scratch')
    argparser.add_argument("--infer-mode", type=int, default=0, help="infer mode")
    argparser.add_argument("--noise", action='store_true', help='noisy dataset')
    argparser.add_argument("--noise_config", default=None, help='noise config')
    argparser.add_argument("--dir", default='chestx/', help='folder to store results')
    

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    if args.noise and not args.eval_only:
        train_data, val_data, test_data = get_noisy_data_loaders(corruption_config=args.noise_config, batch_size=64)
    elif args.noise and args.eval_only:
        _, _, test_data = get_noisy_data_loaders(corruption_config=args.noise_config, apply_to_train=False, apply_to_val=False, batch_size=64)
    else:
        train_data, val_data, test_data = get_data(64)
    # Init Model
    model = DynMMNet(pretrain=1-args.no_pretrain, freeze=args.freeze, directory=args.dir)
    filename = os.path.join('./log', args.dir, 'DynMMNet_freeze_q' + str(args.freeze) + '_reg_' + str(args.reg) + '_noise_' + str(args.noise_config) + '.pt')

    if not args.eval_only:
        model.hard_gate = args.hard
        train(None, None, None, train_data, val_data, args.n_epochs, task="multilabel", optimtype=torch.optim.AdamW,
                is_packed=False, early_stop=True, lr=args.lr, save=filename, weight_decay=args.wd,
                objective=torch.nn.BCEWithLogitsLoss(pos_weight=compute_pos_weights(train_data)), moe_model=model, additional_loss=True, lossw=args.reg)

    # Test
    print(f"Testing model {filename}:")
    model = torch.load(filename).cuda()
    model.hard_gate = True

    # print('-' * 30 + 'Val data' + '-' * 30)
    model.infer_mode = args.infer_mode
    # tmp = test(model=model, test_dataloaders_all=validdata, dataset=args.data, is_packed=False,
    #            criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel", no_robust=True, additional_loss=True)

    print('-' * 30 + 'Test data' + '-' * 30)
    model.reset_weight()
    model.reset_selection_stats()
    model.eval()
    tmp = test(model=model, test_dataloaders_all=test_data, dataset=args.data, is_packed=False,
                criterion=torch.nn.BCEWithLogitsLoss(pos_weight=compute_pos_weights(test_data)), task="multilabel", no_robust=True, additional_loss=True)
    print(model.get_selection_stats())
    print(model.weight_stat())
    print(tmp['f1_micro'], tmp['f1_macro'], model.cal_flop())

