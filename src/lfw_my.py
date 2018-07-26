"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet_my as facenet
DEBUG = True
SHORT_TEST = True

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    if DEBUG:
        print('thresholds = ', thresholds)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    if DEBUG:
        print('embeddings1 = ', embeddings1)
        print('embeddings2 = ', embeddings2)
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    if DEBUG:
        print('next thresholds = ', thresholds)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    if DEBUG:
        print('val, val_std, far = ', val, val_std, far)
    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    i = 0
    for pair in pairs:
        if SHORT_TEST:
            if i > 99:
                break
            i+=1
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
            if False:
                print('len(pair) == 3')
                print('path0 = ', path0)
                print('path1 = ', path1)
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
            if False:
                print('len(pair) == 4')
                print('path0 = ', path0)
                print('path1 = ', path1)
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list
  
def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        if DEBUG:
            print('pairs_filename = ', pairs_filename)
        first = True
        second = True
        i = 0
        for line in f.readlines()[1:]:
            if SHORT_TEST:
                if i >99:
                    break
                i+=1
            pair = line.strip().split()
            if len(pair) == 4 and first:
                first = False
                print('line = ', line)
                print('pair in pairs ', pair)
            if len(pair) == 3 and second:
                second = False
                print('line = ', line)
                print('pair in pairs ', pair)
            pairs.append(pair)
    return np.array(pairs)



