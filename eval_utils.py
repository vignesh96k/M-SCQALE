import argparse
import os 
import json
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

def cov(tensor, rowvar=False, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()
def parse_option():

    parser = argparse.ArgumentParser('argument for testing')
    parser.add_argument('--device', type=int,
                        default=0, help='Device')
    parser.add_argument('--loadpatches', action='store_true',
                        help='Load patches from h5 file')

    parser.add_argument('--loadpatchespath', type=str,
                        default='./SID+LOL_refs_p0.3_pc0.8.npy', help='Path of preselected patches h5')

    parser.add_argument('--ps', type=int,
                        default=96, help='Patch size')    

    parser.add_argument('--p', type=float,
                        default=0.3, help='Sharpness threshold for selecting pristine patches')

    parser.add_argument('--pc', type=float,
                        default=0.8, help='Colorfulness threshold for selecting pristine patches')   

    parser.add_argument('--modelpath', type=str,
                        default='./M-SCQALE_pretrained/', help='Path containing models')

    parser.add_argument('--epochno', type=int,
                        default=110, help='epoch number to be evaluated')

    parser.add_argument('--evalimgspath', type=str,
                        default='./DSLR/', help='Path containing test images')

    parser.add_argument('--refspath', type=str,
                        default='./refs/', help='Path containing images used to select pristine patches')

    parser.add_argument('--outpath', type=str,
                        default='./EVAL_out/', help='output path')

      

   
    optn = parser.parse_args()
    
    optn.saveloc = optn.outpath

    os.makedirs(optn.saveloc, exist_ok=True)
    with open(optn.saveloc + '/config.txt', 'w') as f:
        json.dump(optn.__dict__, f, indent=2)
    

    return optn




def pyrReduce(im,device):
    kernelh = Variable(torch.tensor([[[[0.0625, 0.2500, 0.3750, 0.2500, 0.0625]]]], device=device), requires_grad=False)
    kernelv = Variable(torch.tensor([[[[0.0625], [0.2500], [0.3750], [0.2500], [0.0625]]]], device=device), requires_grad=False)

    kernel = kernelv*kernelh*4
    kernel1 = kernelv*kernelh

    
    if im.size(3) % 2 != 0:
        im = torch.cat((im,im[:,:,:,-1:]),dim=-1)
    if im.size(2) % 2 !=0:
        im = torch.cat((im,im[:,:,-1:,:]),dim=-2)              
    
    
    im_out = torch.zeros(im.size(0),3,int(im.size(2)/2),int(im.size(3)/2), device=device)
    
   
    for k in range(3):
        
        temp = im[:,k,:,:].unsqueeze(dim=1)
        
        im_cp = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
        im_cp = torch.cat((im_cp, im_cp[:,:,:,-1].unsqueeze(dim=3), im_cp[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
        
        im_bp = torch.cat((im_cp[:,:,0,:].unsqueeze(dim=2), im_cp[:,:,0,:].unsqueeze(dim=2), im_cp), dim=2) # padding columns
        im_bp = torch.cat((im_bp, im_bp[:,:,-1,:].unsqueeze(dim=2), im_bp[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
        
        im1 = F.conv2d(im_bp, kernel1, padding = [0,0], groups=1)
        im_out[:,k,:,:] = im1[:,:,0::2,0::2]
    

         
    
    
    return im_out                 


def pyrExpand(im,device):
    
    kernelh = Variable(torch.tensor([[[[0.0625, 0.2500, 0.3750, 0.2500, 0.0625]]]], device=device), requires_grad=False)
    kernelv = Variable(torch.tensor([[[[0.0625], [0.2500], [0.3750], [0.2500], [0.0625]]]], device=device), requires_grad=False)

    kernel = kernelv*kernelh*4
    kernel1 = kernelv*kernelh


    ker00 = kernel[:,:,0::2,0::2]
    ker01 = kernel[:,:,0::2,1::2]
    ker10 = kernel[:,:,1::2,0::2]
    ker11 = kernel[:,:,1::2,1::2]

    
    out = torch.zeros(im.size(0),im.size(1),im.size(2)*2,im.size(3)*2,device=device,dtype=torch.float32)
    
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
def gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size).to(dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def select_patches(all_patches,device,p):
    selected_patches = torch.empty(1, all_patches.size(
        1), all_patches.size(2), all_patches.size(3))
    selected_patches = selected_patches.cuda(device)

    kernel_size = 7
    kernel_sigma = float(7 / 6)
    deltas = []
    for ix in range(all_patches.size(0)):
        rest = all_patches[ix, :, :, :]
        rest = rest.unsqueeze(dim=0)
        rest = transforms.Grayscale()(rest)
        kernel = gaussian_filter(kernel_size=kernel_size, sigma=kernel_sigma).view(
            1, 1, kernel_size, kernel_size).to(rest)
        C = 1
        mu = F.conv2d(rest, kernel, padding=kernel_size // 2)
        mu_sq = mu ** 2
        std = F.conv2d(rest ** 2, kernel, padding=kernel_size // 2)
        std = ((std - mu_sq).abs().sqrt())
        delta = torch.sum(std)
        deltas.append([delta])
    peak_sharpness = max(deltas)[0].item()
    for ix in range(all_patches.size(0)):
        tempdelta = deltas[ix][0].item()
        if tempdelta > p*peak_sharpness:
            selected_patches = torch.cat(
                (selected_patches, all_patches[ix, :, :, :].unsqueeze(dim=0)))
    selected_patches = selected_patches[1:, :, :, :]
    return selected_patches


def select_colorful_patches(all_patches,device,pc):
    selected_patches = torch.empty(1, all_patches.size(
        1), all_patches.size(2), all_patches.size(3))
    selected_patches = selected_patches.cuda(device)
    deltas = []
    for ix in range(all_patches.size(0)):
        rest = all_patches[ix, :, :, :]
        R = rest[0, :, :]
        G = rest[1, :, :]
        B = rest[2, :, :]
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R + G) - B)
        rbMean = torch.mean(rg)
        rbStd = torch.std(rg)
        ybMean = torch.mean(yb)
        ybStd = torch.std(yb)
        stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))

        delta = stdRoot + meanRoot
        deltas.append([delta])
    peak_sharpness = max(deltas)[0].item()
    for ix in range(all_patches.size(0)):
        tempdelta = deltas[ix][0].item()
        if tempdelta > pc*peak_sharpness:
            selected_patches = torch.cat(
                (selected_patches, all_patches[ix, :, :, :].unsqueeze(dim=0)))
    selected_patches = selected_patches[1:, :, :, :]
    return selected_patches