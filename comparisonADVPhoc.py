#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:20:15 2018

@author: akram
"""

import sklearn.metrics.pairwise as skl
from collections import OrderedDict
import os, os.path
import csv
import re
import operator
import ast
import os 
import os.path as osp
from sklearn.metrics import average_precision_score
import numpy as np
from operator import itemgetter

class NotADirectoryError(Exception):
    pass
class FileNotFoundError(Exception):
    pass

def unique(seq):
    seen = set()
    return [seen.add(x) or x for x in seq if x not in seen]


def doubleCheck2():
    
    gt_dict = {}
    #1 create a sorted vector of imgids
    dictImgs = {'100007.jpg'}
    with open("/home/akram/Documents/img_txt/ads_parallelity_dataset.csv", "rb") as file:
        #ignore header of csv file
        next(file)
 
        reader = csv.reader(file)
        for row in reader:
            word_list = []
            img_id = row[0]
            img_id = img_id.replace('.png', '.jpg')
            if img_id != '159028.jpg' and img_id != '99944.jpg':
                dictImgs.add(img_id.strip())
    
    sortedVect = sorted(dictImgs)

    file = open('/home/akram/Documents/img_txt/result.txt', 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    

    for index, line in enumerate(lines):
        #print(line)
        word_list = line.split(":")
        keyLine = word_list[0]
        #print keyLine
        imgs = word_list[1]
        imgarray = ast.literal_eval(imgs)
        if keyLine == 'dos':
            print imgarray
        queryVect = np.zeros(len(sortedVect))
        
        for i, val1 in enumerate(imgarray):
            for j, val2 in enumerate(sortedVect):
                if val1 == val2 :
                    queryVect[j] = 1
        if keyLine == 'dos':
            print imgarray
            print queryVect            
        gt_dict[keyLine.strip()] = queryVect

def doubleCheck():
    #1 create a sorted vector of imgids
    dictImgs = {'100007.jpg'}
    with open("/home/akram/Documents/img_txt/ads_parallelity_dataset.csv", "rb") as file:
        #ignore header of csv file
        next(file)
 
        reader = csv.reader(file)
        for row in reader:
            word_list = []
            img_id = row[0]
            img_id = img_id.replace('.png', '.jpg')
            if img_id != '159028.jpg' and img_id != '99944.jpg':
                dictImgs.add(img_id.strip())
    
    
    file = open('/home/akram/Documents/img_txt/result/added.txt', 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    foundImg = []

    for index, line in enumerate(lines):
        word_list = line.split(":")
        imgId = word_list[0]
        foundImg.append(imgId.strip())
        
        
    #print sorted(dictImgs)
    
    print len(dictImgs)
    print len(foundImg) 
    
    #print sorted(foundImg)    
        
    s = set(dictImgs)
    print s
    temp3 = s - set(foundImg)
    
    #temp3 = [x for x in foundImg if x not in s]
    print temp3
    print len(temp3)
    
def createGrTruthDict():
    
    gt_dict = {}
    gt_score_dict = {}
    #1 create a sorted vector of imgids
    dictImgs = {'100007.jpg'}
    with open("/home/akram/Documents/img_txt/ads_parallelity_dataset.csv", "rb") as file:
        #ignore header of csv file
        next(file)
 
        reader = csv.reader(file)
        for row in reader:
            #word_list = []
            img_id =  row[0]
            img_id = img_id.replace('.png', '.jpg')
            #set(['159028.jpg', '99944.jpg'])
            if img_id != '159028.jpg' and img_id != '99944.jpg':
                dictImgs.add(img_id.strip())
            
    sortedVect = sorted(dictImgs)
    #2 per every word create a sorted zero vector by imgvector size and set the index of that imge which founded in result by one value
    file = open('/home/akram/Documents/img_txt/result.txt', 'r')

    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
        
    for index, line in enumerate(lines):
        #print(line)
        word_list = line.split(":")
        keyLine = word_list[0]
        imgs = word_list[1]
        imgarray = ast.literal_eval(imgs)
        queryVect = np.zeros(len(sortedVect))
        for i, val1 in enumerate(imgarray):
            for j, val2 in enumerate(sortedVect):
                if val1 == val2 :
                    queryVect[j] = 1

        gt_dict[keyLine.strip()] = queryVect
           
    #3 per every word detected sort results in img sorted vector value
        # open the result folder, 
        # per word get the file, 
        # extract image id and distance
        # create sorted result vector with distance values
        
    average = np.zeros(np.shape(sortedVect)[0]) 
    print(len(gt_dict))
    print(len(dictImgs))
    resultDir = '/home/akram/Documents/img_txt/result/'    
    
    try:
        resultlist = [osp.join(osp.realpath('.'), resultDir, txtFile) for txtFile in os.listdir(resultDir)]
    except NotADirectoryError:
        resultlist = []
        resultlist.append(osp.join(osp.realpath('.'), resultDir))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(resultDir))
        exit()
        
    for txtFile in resultlist:

        txt = open(txtFile, 'r')
        fileName = os.path.basename(txt.name)
        #print fileName
        fileNameArr = fileName.split('.')
        txtKey = fileNameArr[0]
        
        txtScoreVector = {}
        
        lines = txt.read().split('\n')                        # store the lines in a list
        lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
        lines = [x for x in lines if x[0] != '#']              # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
        
        for index, line in enumerate(lines):

            word_list = line.split(":")
            imgId = word_list[0]
            distances = word_list[1]
            #print distances
            disArray = ast.literal_eval(distances)
            #print (disArray[0][0])
            txtScoreVector[imgId] = 1.0 - disArray[0][0]
            
        queryVect = np.zeros(len(sortedVect))
        #print txtScoreVector
        sortedScoreVector = sorted(txtScoreVector)

        #print len(sortedVect)
        #print len(sortedScoreVector)
        #print sortedScoreVector
        
        #print ('-------------------------')
        #print list(sortedScoreVector)
        
        for key, value in enumerate(list(sortedScoreVector)):
            #print key
            #print value
            tmp = txtScoreVector[value]
            queryVect[key] = tmp
            
        
        gt_score_dict[txtKey] = queryVect
 
    
    position = 0
    print len(gt_score_dict)
    print len(gt_dict)
    outFile = open('/home/akram/Documents/img_txt/avg_out_file.txt', 'wa') 
    i = 0
    for key in gt_score_dict:
        if key in gt_dict:
            i += 1    
        else:
            print ('key not exist ', key)
    print ('exists keys count, ' , i)
    txt = []
    for key in gt_score_dict:
        if position < 668 :
            
            average[position] = average_precision_score((gt_dict[key]), (gt_score_dict[key]))
            print ('detected key is : ', key, average[position])
            
            txt.append('detected key is : ')
            txt.append( key)
            txt.append( ' ')
            txt.append(str(average[position]))
            txt.append( '\n ')
            if not average[position] :
                print key
                print gt_dict[key]
                print gt_score_dict[key]
            position += 1
               
    print (average)
    print ('mAP: ', np.mean(average))
    outFile.write(''.join(txt))
    outFile.write('\n')
    outFile.write(str(average))
    outFile.write('\n')
    outFile.write(str(np.mean(average)))     
    outFile.close
    
def extractWordDetectedHistogram(txtFile):
    try:
        
        file = open(txtFile, 'r')
    
        lines = file.read().split('\n')                        # store the lines in a list
        lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
        lines = [x for x in lines if x[0] != '#']              # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
        hisVect = np.zeros(668)
        for index, line in enumerate(lines):
            
            word_list = line.split(":")
            #imgId = word_list[0]
            distances = word_list[1]
            #print distances
            disArray = ast.literal_eval(distances)
            #print (disArray[0][0])
            hisVect[index] = disArray[0][0]
            
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(txtFile))
        exit()
    return hisVect

    
def drawHistogramPlot(txtFile):
    import matplotlib.pyplot as plt
    import numpy as np
   
    x = extractWordDetectedHistogram(txtFile)
    print x
    print ('mean is : ', np.mean(x))
    plt.hist(x, normed=True, bins=30)
    plt.ylabel('Histogram');
            
if __name__ == '__main__':
    
    #doubleCheck2()
    #createGrTruthDict()             
    #doubleCheck()           
    drawHistogramPlot('/home/akram/Documents/img_txt/result/MOst.txt')
#calculate every avg by sklearn

#calculate mean avg by sklearn        