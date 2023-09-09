from torch.utils.data import Dataset
import os
import random
from PIL import Image
from numpy.random import randint
from utils import pyrExpand,pyrReduce,pad_diff
import torch
import torchvision.transforms as transforms





class Corpusloader(Dataset):
    def __init__(self, data_dir, k, lev,lowpass,transform=None):
        filenames = os.listdir(data_dir)
        self.pathnames = [os.path.join(data_dir, f) for f in filenames]
        self.lev = lev
        self.lowpass = lowpass
        self.transform = transform
        self.k = k

    def __len__(self):
        return len(self.pathnames)

    def __getitem__(self, idx):

        folder = self.pathnames[idx]
        imgs = os.listdir(folder)
        ref = [string for string in imgs if "WELLLIT" in string]
        dists = [string for string in imgs if "WELLLIT" not in string]    
        
        
        

        distsampled = random.sample(dists, self.k-1)
        current = ref + distsampled
        curr = current[0]

        curr_img = Image.open(folder + '/' + curr)
        curr_img = self.transform(curr_img)

        first = curr_img.unsqueeze(dim=0)
        if self.lev == 0:
            first_pass = first
        
        elif self.lev == 1:
            first_temp = pyrReduce(first)     
            
            if self.lowpass:
                first_pass = first_temp
                # first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)                
            else:                       
                first_pass = first - pyrExpand(first_temp)

        elif self.lev == 2:
            gpyr2 = pyrReduce(first)
            first_temp = pyrReduce(gpyr2)            
            if self.lowpass:
                first_pass = first_temp
                # first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)                
            else:                       
                first_pass = gpyr2 - pyrExpand(first_temp)
    
        elif self.lev == 3:
            gpyr2 = pyrReduce(first)
            gpyr3 = pyrReduce(gpyr2)
            first_temp = pyrReduce(gpyr3)
            if self.lowpass:
                first_pass = first_temp
                first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)                
            else:                       
                first_pass = gpyr3 - pyrExpand(first_temp)

    
        elif self.lev==4:
            gpyr2 = pyrReduce(first)
            gpyr3 = pyrReduce(gpyr2)
            gpyr4 = pyrReduce(gpyr3)
            first_temp = pyrReduce(gpyr4)
            term1 = gpyr4
            term2 = pyrExpand(first_temp)
            if self.lowpass:
                first_pass = first_temp
                
                first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)
                first_pass = torch.cat((first_pass,first_pass[:,:,-1:,:]),dim=-2)    
                
            else:         
            
                first_pass = pad_diff(term1,term2)
    
        elif self.lev==5:
            gpyr2 = pyrReduce(first)
            gpyr3 = pyrReduce(gpyr2)
            gpyr4 = pyrReduce(gpyr3)
            gpyr5 = pyrReduce(gpyr4)
            first_temp = pyrReduce(gpyr5)            
            term1 = gpyr5
            term2 = pyrExpand(first_temp)
            if self.lowpass:
                first_pass = first_temp
            else:
                first_pass = pad_diff(term1,term2)
        


        




        
        for q in range(1, len(current)):
            curr = current[q]
            curr_img = Image.open(folder + '/' + curr)
            curr_img = self.transform(curr_img)

            second = curr_img.unsqueeze(dim=0)

            if self.lev==0:
                second_pass = second
        
            elif self.lev==1:
                second_temp = pyrReduce(second) 
        
                
                if self.lowpass:
                    second_pass = second_temp
                # first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)                
                else:                       
                    second_pass = second - pyrExpand(second_temp)

            elif self.lev==2:
                gpyr2 = pyrReduce(second)
                second_temp = pyrReduce(gpyr2)            
                if self.lowpass:
                    second_pass = second_temp
                # first_pass = torch.cat((first_pass,first_pass[:,:,:,-1:]),dim=-1)                
                else:                       
                    second_pass = gpyr2 - pyrExpand(second_temp)
                
        
            elif self.lev==3:
                gpyr2 = pyrReduce(second)
                gpyr3 = pyrReduce(gpyr2)
                second_temp = pyrReduce(gpyr3)            
                if self.lowpass:
                    second_pass = second_temp
                    second_pass = torch.cat((second_pass,second_pass[:,:,:,-1:]),dim=-1)
                else:                       
                    second_pass = gpyr3 - pyrExpand(second_temp)
                   
        
            elif self.lev==4:
                gpyr2 = pyrReduce(second)
                gpyr3 = pyrReduce(gpyr2)
                gpyr4 = pyrReduce(gpyr3)
                second_temp = pyrReduce(gpyr4)            
                term1 = gpyr4
                term2 = pyrExpand(second_temp)    
                if self.lowpass:
                    second_pass = second_temp
                    second_pass = torch.cat((second_pass,second_pass[:,:,:,-1:]),dim=-1)
                    second_pass = torch.cat((second_pass,second_pass[:,:,-1:,:]),dim=-2)
                else:                       
                    second_pass = pad_diff(term1,term2)
            elif self.lev==5:
                gpyr2 = pyrReduce(second)
                gpyr3 = pyrReduce(gpyr2)
                gpyr4 = pyrReduce(gpyr3)
                gpyr5 = pyrReduce(gpyr4)
                second_temp = pyrReduce(gpyr5)            
                term1 = gpyr5
                term2 = pyrExpand(second_temp)
                if self.lowpass:
                    second_pass = second_temp
                else:
                                        
                    second_pass = pad_diff(term1,term2)
            
            
            first_pass = torch.cat((first_pass, second_pass))

        batch = first_pass
        ps = min(int(batch.size(2)/2),int(batch.size(3)/2))

        ###twopatchsampler############
        f = randint(1, 3)
        if f == 1:
            half1, half2 = torch.split(
                batch, int(batch.size(2)/2), dim=2)            

        else:
            half1, half2 = torch.split(
                batch, int(batch.size(3)/2), dim=3)

        crop = transforms.RandomCrop(ps)
        p1_batch = crop(half1)
        p2_batch = crop(half2)

        return p1_batch, p2_batch