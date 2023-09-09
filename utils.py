import argparse
import os 
import json
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--level', type=int,
                        default=0, help='Level')
    # parser.add_argument('--ngpus', type=int,
    #                     default=2, help='no of gpus')
    parser.add_argument('--device', type=int,
                        default=0, help='Device')
    parser.add_argument('--k', type=int,
                        default=10, help='negative samples')
    parser.add_argument('--bs', type=int,
                        default=4, help='no of scenes perbatch')
    parser.add_argument('--tao', type=float,
                        default=0.1, help='temperature')      
    parser.add_argument('--epochs', type=int,
                        default=110, help='no of epochs')
    parser.add_argument('--savefolder', type=str,
                        default='./checkpoints_NEWD_parallel/', help='savefolder')
    parser.add_argument('--traindatapath', type=str,
                        default='/media/ece/a4ceab9a-5891-4860-a7c6-a04cf86651de/Vignesh/VIGNESH_DATASET/Data/NEW_DATASET/', help='traindatapath')
    parser.add_argument('--save_freq', type=int,
                        default=1, help='Save model at every nth epoch')    

    parser.add_argument('--lowpass', action='store_true',
                        help='use current level lowpass')
    optn = parser.parse_args()
    if optn.lowpass:             
        optn.saveloc = optn.savefolder  + 'level_' + str(optn.level) + '_lowpass'
    else:
        optn.saveloc = optn.savefolder  + 'level_' + str(optn.level)  
    

    os.makedirs(optn.saveloc, exist_ok=True)
    with open(optn.saveloc + '/config.txt', 'w') as f:
        json.dump(optn.__dict__, f, indent=2)
    

    return optn





def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))

    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


kernelh = Variable(torch.tensor([[[[0.0625, 0.2500, 0.3750, 0.2500, 0.0625]]]]), requires_grad=False)
kernelv = Variable(torch.tensor([[[[0.0625], [0.2500], [0.3750], [0.2500], [0.0625]]]]), requires_grad=False)

kernel = kernelv*kernelh*4
kernel1 = kernelv*kernelh

ker00 = kernel[:,:,0::2,0::2]
ker01 = kernel[:,:,0::2,1::2]
ker10 = kernel[:,:,1::2,0::2]
ker11 = kernel[:,:,1::2,1::2]

def BuildLapPyr(im):
    gpyr2 = pyrReduce(im)
    gpyr3 = pyrReduce(gpyr2)
    gpyr4 = pyrReduce(gpyr3)
    # gpyr5 = pyrReduce(gpyr4)
    
    sub1 = im - pyrExpand(gpyr2)
    sub2 = gpyr2 - pyrExpand(gpyr3)
    sub3 = gpyr3 - pyrExpand(gpyr4)
    
        
    # return sub1, sub2, sub3, sub4 ,gpyr5
    return sub1, sub2, sub3, gpyr4

def pyrReduce(im):
    if im.size(3) % 2 != 0:
        im = torch.cat((im,im[:,:,:,-1:]),dim=-1)
    if im.size(2) % 2 !=0:
        im = torch.cat((im,im[:,:,-1:,:]),dim=-2)              
    
    
    im_out = torch.zeros(im.size(0),3,int(im.size(2)/2),int(im.size(3)/2))
    
   
    for k in range(3):
        
        temp = im[:,k,:,:].unsqueeze(dim=1)
        
        im_cp = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
        im_cp = torch.cat((im_cp, im_cp[:,:,:,-1].unsqueeze(dim=3), im_cp[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
        
        im_bp = torch.cat((im_cp[:,:,0,:].unsqueeze(dim=2), im_cp[:,:,0,:].unsqueeze(dim=2), im_cp), dim=2) # padding columns
        im_bp = torch.cat((im_bp, im_bp[:,:,-1,:].unsqueeze(dim=2), im_bp[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
        
        im1 = F.conv2d(im_bp, kernel1, padding = [0,0], groups=1)
        im_out[:,k,:,:] = im1[:,:,0::2,0::2]
    

         
    
    
    return im_out                 


def pyrExpand(im):
    


    
    out = torch.zeros(im.size(0),im.size(1),im.size(2)*2,im.size(3)*2, dtype=torch.float32)
    
    for k in range(3):
        
        temp = im[:,k,:,:]
        temp = temp.unsqueeze(dim=1)
                       
        im_c1 = torch.cat((temp, temp[:,:,:,-1].unsqueeze(dim=3)), dim=3) 
        im_c1r1 = torch.cat((im_c1, im_c1[:,:,-1,:].unsqueeze(dim=2)), dim=2) 
                
        im_r2 = torch.cat((temp[:,:,0,:].unsqueeze(dim=2), temp), dim=2) # padding columns
        im_r2 = torch.cat((im_r2, im_r2[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
        
        im_r2c1 = torch.cat((im_r2, im_r2[:,:,:,-1].unsqueeze(dim=3)), dim=3) 
                
        im_c2 = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
        im_c2 = torch.cat((im_c2, im_c2[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
        
        im_c2r1 = torch.cat((im_c2, im_c2[:,:,-1,:].unsqueeze(dim=2)), dim=2) 
                
        im_r2c2 = torch.cat((im_c2[:,:,0,:].unsqueeze(dim=2), im_c2), dim=2) # padding columns
        im_r2c2 = torch.cat((im_r2c2, im_r2c2[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
                
        im_00 = F.conv2d(im_r2c2, ker00, padding = [0,0], groups=1)
        im_01 = F.conv2d(im_r2c1, ker01, padding = [0,0], groups=1)
        im_10 = F.conv2d(im_c2r1, ker10, padding = [0,0], groups=1)
        im_11 = F.conv2d(im_c1r1, ker11, padding = [0,0], groups=1)
        
        out[:,k,0::2,0::2] = im_00
        out[:,k,1::2,0::2] = im_10
        out[:,k,0::2,1::2] = im_01
        out[:,k,1::2,1::2] = im_11
                 
    return out

def pad_diff(term1,term2):
    max_h = max([term1.size(-2),term2.size(-2)])
    max_w = max([term1.size(-1),term2.size(-1)])
    w_diff1 = max_w - term1.size(-1)
    h_diff1 = max_h - term1.size(-2)
    if w_diff1 != 0 :
        for x in range(w_diff1):
            term1 = torch.cat((term1,term1[:,:,:,-1:]),dim=-1)
    if h_diff1 != 0 :
        for x in range(h_diff1):
            term1 = torch.cat((term1,term1[:,:,-1:,:]),dim=-2)

    w_diff2 = max_w - term2.size(-1)
    h_diff2 = max_h - term2.size(-2)
    if w_diff2 != 0 :
        for x in range(w_diff2):
            term2 = torch.cat((term2,term2[:,:,:,-1:]),dim=-1)
    if h_diff2 != 0 :
        for x in range(h_diff2):
            term2 = torch.cat((term2,term2[:,:,-1:,:]),dim=-2)

    diff = term1 - term2
    return diff

def createindmat(k, bs):

    liste = []
    bs = int(k)
    views = 2
    for i in range(1, bs+1):
        for j in range(1, views+1):
            temp = 'image' + str(i) + '_' + 'view' + str(j)
            liste.append(temp)
    A = np.array(liste, object)
    B = np.array(liste, object)
    B = B.T

    C = A[None, :] + '_' + B[:, None]
    C = C.T

    return C


