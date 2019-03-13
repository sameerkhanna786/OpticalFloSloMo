import argparse
import os
import os.path
from shutil import rmtree, move
import random
import shutil

dataset_folder = "dataset"
videos_folder = "Adobe"
imgwidth = "640"
imgheight = "360"
train_test = (90, 10)

def extract_images(video_list, inputDir, outputDir):
    for video in video_list:
        inTemp = os.path.join(inputDir, video)
        outTemp = os.path.join(outputDir, os.path.splitext(video)[0])
        if not os.path.isdir(outTemp):
            os.mkdir(outTemp)
        SytemCom = 'ffmpeg.exe' + ' -i ' + inTemp + ' -vf scale=' + imgwidth + ':' + imgheight + ' -vsync 0 -qscale:v 2 ' + outTemp + '/%04d.jpg'
        error = os.system(SytemCom)
        if error:
            error_msg = "Error with file: " + video
            print(error_msg)

def combine_images(inputDir, outputDir, clip_size = 12):
    folderNum = -1
    file_list = os.listdir(inputDir)
    for file in file_list:
        imageLoc = os.path.join(inputDir, file)
        image_list = os.listdir(imageLoc)
        for imageNum, image in enumerate(image_list):
            if (imageNum % clip_size == 0):
                if (imageNum + clip_size - 1 >= len(image_list)):
                    break
                folderNum += 1
                folderName = os.path.join(outputDir, str(folderNum))
                os.mkdir(folderName)
            folderName = os.path.join(outputDir, str(folderNum))
            imageName = os.path.join(folderName, image)
        rmtree(imageLoc)

#create the necessary missing directories
lst = []
extractPath      = os.path.join(dataset_folder, "extracted")
trainPath        = os.path.join(dataset_folder, "train")
testPath         = os.path.join(dataset_folder, "test")
validationPath   = os.path.join(dataset_folder, "validation")       
lst.append(dataset_folder)
lst.append(extractPath)
lst.append(trainPath)
lst.append(testPath)
lst.append(validationPath)
for i in lst:
    if not os.path.isdir(i):
        os.mkdir(i)
        
#generate list of all the videos
video_list = os.listdir(videos_folder)

#shuffle list of videos to determine which goes into training and testing batches
random.shuffle(video_list)

#create the lists for training and testing
testNames = video_list[:train_test[1]]
trainNames = video_list[train_test[1]:]

#Create the training and testing dataset
extract_images(testNames, videos_folder, extractPath)
combine_images(extractPath, testPath)
extract_images(trainNames, videos_folder, extractPath)
combine_images(extractPath, trainPath)

#select clips randomly for validation
testClips = os.listdir(testPath)
indices = random.sample(range(len(testClips)), min(100, int(len(testClips) / 5)))
for index in indices:
    move("{}/{}".format(testPath, index), "{}/{}".format(validationPath, index))

#remove the files in the extracted path        
rmtree(extractPath)
