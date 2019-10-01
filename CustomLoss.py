import torch
import torch.nn as nn
import numpy as np


# optimized weighted binary cross entropy loss
class Loss(nn.modules.Module):
    def __init__(self, train_df, valid_df, device):
        super(Loss, self).__init__()
        
        total_positive_images_train = (train_df.Label == 1).sum()
        total_negative_images_train = (train_df.Label == 0).sum()
        Wt1_train = total_negative_images_train/(total_negative_images_train + total_positive_images_train)
        Wt0_train = total_positive_images_train/(total_negative_images_train + total_positive_images_train)

        total_positive_images_valid = (valid_df.Label == 1).sum()
        total_negative_images_valid = (valid_df.Label == 0).sum()
        Wt1_valid = total_negative_images_valid/(total_negative_images_valid + total_positive_images_valid)
        Wt0_valid = total_positive_images_valid/(total_negative_images_valid + total_positive_images_valid)

        Wt = dict()
        Wt_train={}
        Wt_valid={}
        Wt_train['Wt1'] = torch.from_numpy(np.asarray(Wt1_train)).double().type(torch.FloatTensor).to(device)
        Wt_train['Wt0'] = torch.from_numpy(np.asarray(Wt0_train)).double().type(torch.FloatTensor).to(device)
        Wt_valid['Wt1'] = torch.from_numpy(np.asarray(Wt1_valid)).double().type(torch.FloatTensor).to(device)
        Wt_valid['Wt0'] = torch.from_numpy(np.asarray(Wt0_valid)).double().type(torch.FloatTensor).to(device)

        Wt['train'] = Wt_train
        Wt['valid'] = Wt_valid
        
        self.Wt = Wt

    def forward(self, inputs, targets, cat):
        loss = -(self.Wt[cat]['Wt1'] * targets * inputs.log() + self.Wt[cat]['Wt0'] * (1 - targets) * (1 - inputs).log())
        return loss.mean()
