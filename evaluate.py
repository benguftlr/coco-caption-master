#from dask.compatibility import FileExistsError

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='.'
dataType='val2017'
# algName = 'fakecap'
subtype='results'
annFile='annotations/references_2.json'
# resFile='results/captions_val2017_cp_0005.json'
# resFile='%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype)
# subtypes=['results', 'evalImgs', 'eval']
# [resFile, evalImgsFile, evalFile]= \
# ['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]

# download Stanford models
# !./get_stanford_models.sh

# create coco object and cocoRes object
coco = COCO(annFile)
# cocoRes = coco.loadRes(resFile)
#
#
# # create cocoEval object by taking coco and cocoRes
# cocoEval = COCOEvalCap(coco, cocoRes)
#
# # evaluate on a subset of images by setting
# # cocoEval.params['image_id'] = cocoRes.getImgIds()
# # please remove this line when evaluating the full validation set
# cocoEval.params['image_id'] = cocoRes.getImgIds()
#
# # evaluate results
# # SPICE will take a few minutes the first time, but speeds up due to caching
# cocoEval.evaluate()


import glob
from tqdm import tqdm
import os

dirName = 'results'


import time as t
json_file = glob.glob("results/predicted_2.json")

# Initialize the whole list as empty list
resultList = []

# Append titles
nameList = ["MODEL_NAME","CIDEr", "BLEU-4", "BLEU-3", "BLEU-2", "BLEU-1", "ROUGE_L", "METEOR", "SPICE"]
resultList.append(nameList)

# Start evaluating all json files in the results folder.
for resFile in tqdm(json_file):

    # Get json name
    name = resFile.split("/")
    name_with_extension = name[1]
    json_name = name_with_extension[:-5]
    print(json_name)
    # Refresh the list for the model
    list = []
    # Append the json's model name
    list.append(json_name)
    # print output evaluation scores
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # Append scores to the list
    for metric, score in cocoEval.eval.items():
        print ('%s: %.3f' % (metric, score))
        list.append(score)

    # Append scores to the whole list
    resultList.append(list)

print (resultList)
# Write results into a csv file
import csv
with open("result.csv", 'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(resultList)


# import pandas as pd
# df_data_4 = pd.read_csv('dashdb.csv')
# df_data_4.head()
#
#
# # demo how to use evalImgs to retrieve low score result
# evals = [eva for eva in cocoEval.evalImgs if eva['CIDEr']<30]
# print 'ground truth captions'
# imgId = evals[0]['image_id']
# annIds = coco.getAnnIds(imgIds=imgId)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
#
# print '\n'
# print 'generated caption (CIDEr score %0.1f)'%(evals[0]['CIDEr'])
# annIds = cocoRes.getAnnIds(imgIds=imgId)
# anns = cocoRes.loadAnns(annIds)
# coco.showAnns(anns)
#
# img = coco.loadImgs(imgId)[0]
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# plt.imshow(I)
# plt.axis('off')
# plt.show()
#
# # plot score histogram
# ciderScores = [eva['CIDEr'] for eva in cocoEval.evalImgs]
# plt.hist(ciderScores)
# plt.title('Histogram of CIDEr Scores', fontsize=20)
# plt.xlabel('CIDEr score', fontsize=20)
# plt.ylabel('result counts', fontsize=20)
# plt.show()
#
#
# # save evaluation results to ./results folder
# json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
# json.dump(cocoEval.eval,     open(evalFile, 'w'))
