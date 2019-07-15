import torch
from PIL import Image
import numpy as np

class MuraDataset(torch.utils.data.Dataset):
    
    def __init__(self,df,transform=None):
        self.df=df
        self.transform=transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        img_name=self.df.iloc[idx,0]
        img=Image.open(img_name).convert('LA')
        label=self.df.iloc[idx,1]

        if self.transform:
            img=self.transform(img)
        label = torch.from_numpy(np.asarray(label)).double().type(torch.FloatTensor)
        return img, label

