from losses import ContrastiveLoss
from utils import parse_option
import os
import time
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import resnet50
from utils import *
from datasets import Corpusloader


args = parse_option()


k = args.k
tao = args.tao
data_dir = args.traindatapath
save_freq = args.save_freq
bs = args.bs
epochs = args.epochs
# devices = list(range(args.ngpus))
cuda1 = torch.device('cuda:' + str(args.device))
transform_train = transforms.Compose([
    
    transforms.ToTensor()
])
train_dataloader = DataLoader(Corpusloader(data_dir, k, args.level,args.lowpass,transform_train),
                              batch_size=bs, shuffle=True,drop_last=True,num_workers=4)

    
    
res = resnet50()
modules = list(res.children())[:-1]
model = nn.Sequential(*modules)
  

# model = torch.nn.DataParallel(model, device_ids=devices)

indmat = createindmat(k,bs)
model.train()
model = model.cuda(cuda1)
opt = optim.Adam(model.parameters(), lr=1e-2)


criterion = ContrastiveLoss(modes = ['v1ancdisv2','v2ancdisv1'])
eps =1e-8
from tqdm import tqdm
for epoch in range(epochs):
    epoch_loss = 0
    start_time = time.time()
    for n_count, batch in enumerate(tqdm(train_dataloader)):
        batchtime = time.time()
        opt.zero_grad()
        p1_batch, p2_batch = Variable(batch[0]), Variable(
            batch[1])
        
        
                   
        stacked = torch.stack((p1_batch, p2_batch), dim=2)  
        ps = stacked.size(-1)
        stacked = stacked.view(bs*2*k,3,ps,ps)

        passsamples = stacked.cuda(cuda1)

        feats = model(passsamples)
       
        
        feats = feats.view(bs,k,2,-1)
        
       
        interleaved = torch.flatten(feats, start_dim=1, end_dim=2)

        
        Ai = interleaved
        Anorm = torch.linalg.norm(Ai, dim=2)
        a_norm = torch.max(Anorm, eps * torch.ones_like(Anorm))
        A_n = Ai / a_norm.view(bs, 2*k, 1)
        mat = torch.bmm(A_n, torch.transpose(A_n, 1, 2))
        mat = mat/tao               
        loss = criterion(mat,indmat)
        epoch_loss += loss.item()
        loss.backward()        
        opt.step()
        batchtimed = time.time() - batchtime
    elapsed_time = time.time() - start_time
    print('epoch = %4d , loss = %4.4f , time = %4.2f s' %
          (epoch + 1, epoch_loss / n_count, elapsed_time))
    if (epoch + 1) % save_freq == 0:
        torch.save(model, os.path.join(
            args.saveloc, 'model_%03d.pth' % (epoch+1)))
