import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from scipy.stats import spearmanr
from PIL import Image
from sklearn.decomposition import PCA
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from tqdm import tqdm
from scipy.io import savemat
import os
from eval_utils import *
args = parse_option()

cuda1 = torch.device('cuda:' + str(args.device))

data_path = args.evalimgspath + 'Images/'
refpath = args.refspath
toten = transforms.ToTensor()
refs = os.listdir(refpath)

outfolder = args.outpath
os.makedirs(outfolder, exist_ok=True)
os.makedirs(outfolder +'/scores/', exist_ok=True)

ref_feats = []
ps = args.ps
p = args.p
pc = args.pc
if args.loadpatches:
    first_patches = np.load(args.loadpatchespath)
    first_patches = torch.tensor(first_patches,device=cuda1)
    print("----------------------Loaded Preselected Patches-------------------------")

else:
    temp = np.array(Image.open(refpath + refs[0]))
    toten = transforms.ToTensor()
    temp = toten(temp)
    batch = temp.cuda(cuda1)
    batch = batch.unsqueeze(dim=0)
    patches = batch.unfold(1, 3, 3).unfold(
        2, ps, ps).unfold(3, ps, ps)

    patches = patches.contiguous().view(1, -1, 3,
                                        ps, ps)

    for ix in range(patches.size(0)):
        patches[ix, :, :, :, :] = patches[ix, torch.randperm(
            patches.size()[1]), :, :, :]
    first_patches = patches.squeeze()
    first_patches = select_colorful_patches(select_patches(first_patches,cuda1,p),cuda1,pc)


    refs = refs[1:]
    print("-----------------Selecting Pristine Patches------------------------")
    for ir,rs in enumerate(tqdm(refs)):
        temp = np.array(Image.open(refpath + rs))
        toten = transforms.ToTensor()
        temp = toten(temp)
        batch = temp.cuda(cuda1)
        batch = batch.unsqueeze(dim=0)
        patches = batch.unfold(1, 3, 3).unfold(
            2, ps, ps).unfold(3, ps, ps)

        patches = patches.contiguous().view(1, -1, 3,
                                            ps, ps)

        for ix in range(patches.size(0)):
            patches[ix, :, :, :, :] = patches[ix, torch.randperm(
                patches.size()[1]), :, :, :]
        second_patches = patches.squeeze()
        second_patches = select_colorful_patches(select_patches(second_patches,cuda1,p),cuda1,pc)
        first_patches = torch.cat((first_patches, second_patches))
        
    first_patches = first_patches.detach().cpu().numpy()
    # np.save('./SID+LOL_refs_p0.3_pc0.8.npy',first_patches,allow_pickle=False) 

    print("----------------------Selected Pristine Patches-------------------------")
    




toten = transforms.ToTensor()
savefolder = args.modelpath
modelp0 = savefolder + 'level_0' + '/model_' + "%03d" % args.epochno + '.pth'
modelp1 = savefolder + 'level_1' + '/model_' + "%03d" % (args.epochno) + '.pth'
modelp2 = savefolder + 'level_2' + '/model_' + "%03d" % (args.epochno) + '.pth'
modelp3 = savefolder + 'level_3' + '/model_' + "%03d" % (args.epochno) + '.pth'
modelp3low = savefolder + 'level_3_lowpass' + '/model_' + "%03d" % (args.epochno) + '.pth'


model0 = torch.load(modelp0,map_location='cpu')
model0.eval()
model0 = model0.cuda(cuda1)


model1 = torch.load(modelp1,map_location='cpu')
model1.eval()
model1 = model1.cuda(cuda1)

model2 = torch.load(modelp2,map_location='cpu')
model2.eval()
model2 = model2.cuda(cuda1)

model3 = torch.load(modelp3,map_location='cpu')
model3.eval()
model3 = model3.cuda(cuda1)

model3low = torch.load(modelp3low,map_location='cpu')
model3low.eval()
model3low = model3low.cuda(cuda1)

levs = 5
all_ref_feats = torch.empty(first_patches.size(0), 2048*levs)

print("--------------------Computing features for Pristine Patches------------------------")
for ix in tqdm( range(first_patches.size(0))):
    rest = first_patches[ix, :, :, :]

    rest = rest.unsqueeze(dim=0)
    gpyr2 = pyrReduce(rest,cuda1)
    gpyr3 = pyrReduce(gpyr2,cuda1)
    gpyr4 = pyrReduce(gpyr3,cuda1)
    # gpyr5 = pyrReduce(gpyr4)

    sub1 = rest - pyrExpand(gpyr2,cuda1)
    sub2 = gpyr2 - pyrExpand(gpyr3,cuda1)
    sub3 = gpyr3 - pyrExpand(gpyr4,cuda1)

    with torch.no_grad():
        feat0 = (model0(rest)).squeeze()
        feat1 = (model1(sub1)).squeeze()
        feat2 = (model2(sub2)).squeeze()
        feat3 = (model3(sub3)).squeeze()
        feat3low = (model3low(gpyr4)).squeeze()
    resfeat = torch.cat((feat0, feat1, feat2, feat3, feat3low), dim=0)

    all_ref_feats[ix, :] = resfeat
    
print("--------------------Computed features for Pristine Patches------------------------")



pca = PCA(2048)
pca.fit(all_ref_feats)
all_ref_feats = pca.transform(all_ref_feats)
all_ref_feats = torch.tensor(all_ref_feats)
vr = torch.mean(all_ref_feats, dim=0)

covr = cov(all_ref_feats)
print("Computed vr and sigmar")
data = pd.read_csv(args.evalimgspath + 'MOS_FR.csv', names=["im_loc", "mos","ref_loc"])
data = data.drop([data.index[0]])
all_names = list(data['im_loc'])
all_mos = list(data['mos'])
scores = []
moss = []

for ind, x in enumerate(tqdm(all_names)):
    if ind > 5:
        break
    times = time.time()
    temp = np.array(Image.open(data_path + x))
    toten = transforms.ToTensor()
    temp = toten(temp)
    batch = temp.cuda(cuda1)
    batch = batch.unsqueeze(dim=0)
    patches = batch.unfold(1, 3, 3).unfold(
        2, ps, int(ps/2)).unfold(3, ps, int(ps/2))

    patches = patches.contiguous().view(1, -1, 3,
                                        ps, ps)

    for ix in range(patches.size(0)):
        patches[ix, :, :, :, :] = patches[ix, torch.randperm(
            patches.size()[1]), :, :, :]
    patches = patches.squeeze()
    all_rest_feats = torch.empty(patches.size(0), 2048*levs)

    for ix in range(patches.size(0)):
        rest = patches[ix, :, :, :]
        rest = rest.unsqueeze(dim=0)
        gpyr2 = pyrReduce(rest,cuda1)
        gpyr3 = pyrReduce(gpyr2,cuda1)
        gpyr4 = pyrReduce(gpyr3,cuda1)

        sub1 = rest - pyrExpand(gpyr2,cuda1)
        sub2 = gpyr2 - pyrExpand(gpyr3,cuda1)
        sub3 = gpyr3 - pyrExpand(gpyr4,cuda1)

        with torch.no_grad():
            feat0 = (model0(rest)).squeeze()
            feat1 = (model1(sub1)).squeeze()
            feat2 = (model2(sub2)).squeeze()
            feat3 = (model3(sub3)).squeeze()
            feat3low = (model3low(gpyr4)).squeeze()
        resfeat = torch.cat((feat0, feat1, feat2, feat3, feat3low), dim=0)
        all_rest_feats[ix, :] = resfeat

    all_rest_feats = pca.transform(all_rest_feats)

    all_rest_feats = torch.tensor(all_rest_feats)
    vd = torch.mean(all_rest_feats, dim=0)
    covd = cov(all_rest_feats)
    ft = vr - vd
    ft = torch.unsqueeze(vr-vd, dim=0)
    st = torch.inverse(((covd + covr) / 2) + 0.001*torch.eye(covd.size(0)))

    tt = torch.transpose(ft, 0, 1)
    imt = torch.matmul(ft, st)
    fit = torch.matmul(imt, tt)
    score = (torch.sqrt(fit)).squeeze()
    score = Variable(score,
                    requires_grad=False).cpu().numpy()

    scores.append(score)
    feat_ind = all_names.index(x)
    moss.append(all_mos[feat_ind])
    savemat(outfolder + '/scores/' + all_names[ind][:-4] +
            '.mat', {'feat': np.array(score)})


rho_test, p_test = spearmanr(np.array(moss), np.array(scores))
print(rho_test)

out = []
out.append(rho_test)
df = pd.DataFrame()
df['SRCC'] = out
df.to_csv(args.outpath + "Results.csv")
