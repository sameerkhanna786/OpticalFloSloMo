#!/usr/bin/env python3
import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import model
import dataloader
import platform
from tqdm import tqdm

input_video = "subaru.mp4"
checkpoint_dir = 'checkpoints/SuperSloMo.ckpt'
fps_out = 30
slow_fac = 4
output_video = "subaru_out.mp4"

def extract_images(video, outputDir):
    sysCmd = 'ffmpeg -i ' + video + ' -vsync 0 -qscale:v 2 ' + outputDir + '/%06d.jpg'
    print(sysCmd)
    os.system(sysCmd)


def compile_video(inputDir):
    sysCmd = 'ffmpeg -r '+fps_out+' -i '+inputDir+'/%d.jpg -qscale:v 2 '+output_video
    print(sysCmd)
    os.system(sysCmd)       

extractionDir = "tmpSuperSloMo"
os.mkdir(extractionDir)
extractionPath = os.path.join(extractionDir, "input")
outputPath = os.path.join(extractionDir, "output")
os.mkdir(extractionPath)
os.mkdir(outputPath)
extract_images(input_video, extractionPath)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)

if (device == "cpu"):
    transform = transforms.Compose([transforms.ToTensor()])
    TP = transforms.Compose([transforms.ToPILImage()])
else:
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

frames = dataloader.Video(root = extractionPath, transform = transform)
frameLoader = torch.utils.data.DataLoader(frames, batch_size = 1, shuffle = False)

flowComp = model.UNet(6, 4)
flowComp.to(device)
timeFlowIntrp = model.UNet(20, 5)
timeFlowIntrp.to(device)
for param in flowComp.parameters():
    param.requires_grad = False
for param in timeFlowIntrp.parameters():
    param.requires_grad = False

backWarp = model.backWarp(frames.dim[0], frames.dim[1], device)
backWarp = backWarp.to(device)

model_dict = torch.load(checkpoint_dir, map_location='cpu')
timeFlowIntrp.load_state_dict(model_dict['state_dictAT'])
flowComp.load_state_dict(model_dict['state_dictFC'])

counter = 1

with torch.no_grad():
    for _, (frame0, frame1) in enumerate(tqdm(frameLoader), 0):
        I0 = frame0.to(device)
        I1 = frame1.to(device)

        outflow = flowComp(torch.cat((I0, I1), dim=1))
        flow_zero_one = outflow[:, :2, :, :]
        flow_one_zero = outflow[:, 2:, :, :]
        
        for batchIndex in range(1):
            frameNum = counter + slow_fac*batchIndex
            (TP(frame0[batchIndex].detach())).resize(frames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameNum) + ".jpg"))
        counter += 1

        for interFNum in range(1, slow_fac):
            t = interFNum / slow_fac
            temp = -t*(1 - t)

            flow_t_zero = temp*flow_zero_one + t*t*flow_one_zero
            flow_t_one = (1 - t)*(1 - t)*flow_zero_one + temp*flow_one_zero

            g_I0_F_t_zero = backWarp(I0, flow_t_zero)
            g_I1_F_t_one = backWarp(I1, flow_t_one)

            intrpOut = timeFlowIntrp(torch.cat((I0, I1, flow_zero_one, flow_one_zero, flow_t_one, flow_t_zero, g_I1_F_t_one, g_I0_F_t_zero), dim=1))

            flow_t_zero_f = intrpOut[:, :2, :, :] + flow_t_zero
            flow_t_one_f = intrpOut[:, 2:4, :, :] + flow_t_one
            V_t_zero = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_one = 1 - V_t_zero
                    
            g_I0_flow_t_zero_f = backWarp(I0, flow_t_zero_f)
            g_I1_flow_t_one_f = backWarp(I1, flow_t_one_f)
                
            wCoeff = [1 - t, t]

            Ft_p = ((1-t) * V_t_zero * g_I0_flow_t_zero_f + t * V_t_one * g_I1_flow_t_one_f) / ((1-t) * V_t_zero + t * V_t_one)

            for batchIndex in range(1):
                frameNum = counter + slow_fac*batchIndex
                (TP(Ft_p[batchIndex].cpu().detach())).resize(frames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameNum) + ".jpg"))
            counter += 1           
create_video(outputPath)
rmtree(extractionDir)














        
