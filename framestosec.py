#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:14:19 2018

@author: jweng
"""

def framesToSec(x):
    newList = []
    for num in x:
        newList.append(num/31.39)

    return newList


#start = [374, 498, 654, 907, 1027, 1171, 1316, 1976, 2103, 2224, 2477, 2607, 2742, 2863]
#end = [394, 563, 671, 923, 1086, 1197, 1339, 2002, 2121, 2271, 2501, 2635, 2777, 2883]

start = [273,298,667,700,1098,1117,1403,1421,2877,2892]

#print(len(start))
#print('Start: ' + str(framesToSec(start)))
#print('End: ' + str(framesToSec(end)))

import os

root_path = './train'
folders = ['cat']

gh = folders
print(gh)
for folder in gh:
    os.mkdir(os.path.join(root_path,str(folder)))
