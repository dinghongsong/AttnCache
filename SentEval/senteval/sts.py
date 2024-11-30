# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging
import random
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from senteval.utils import cosine
from senteval.sick import SICKEval


import numpy as np
from scipy.spatial.distance import jensenshannon

def js_divergence(matrix1, matrix2):
    """
    Calculate the Jensen-Shannon divergence between two matrices.
    
    Parameters:
        matrix1 (numpy.ndarray): The first matrix with shape [1, 24, 128, 128].
        matrix2 (numpy.ndarray): The second matrix with shape [1, 24, 128, 128].
        
    Returns:
        float: The average JS divergence across all dimensions.
    """
    # Flatten along the batch and channel dimensions
    matrix1 = matrix1.reshape(24, -1)
    matrix2 = matrix2.reshape(24, -1)
    
    # # Normalize the matrices along each feature vector (axis=1)
    # matrix1 = matrix1 / matrix1.sum(axis=1, keepdims=True)
    # matrix2 = matrix2 / matrix2.sum(axis=1, keepdims=True)
    
    # Calculate JS divergence for each feature vector
    js_divs = []
    for vec1, vec2 in zip(matrix1, matrix2):
        js_div = jensenshannon(vec1, vec2)  # Returns square root of JS divergence
        js_divs.append(js_div**2)  # Convert to actual JS divergence
    
    # Return the average JS divergence
    return round(np.mean(js_divs), 2)
    # return round(js_divs, 2)

def calculate_js_matrix(matrix_list):
    """
    Calculate the pairwise JS divergence for a list of matrices.
    
    Parameters:
        matrix_list (list): A list of 28 matrices, each with shape [1, 24, 128, 128].
        
    Returns:
        numpy.ndarray: A 28x28 matrix of pairwise JS divergences.
    """
    n = len(matrix_list)
   
    # Initialize the result matrix
    js_matrix = np.zeros((n, n))
    
    # Calculate pairwise JS divergence
    for i in range(n):
        for j in range(i, n):  # Use symmetry to optimize calculations
            js_value = js_divergence(matrix_list[i], matrix_list[j])
            js_matrix[i, j] = js_value
            js_matrix[j, i] = js_value  # Symmetric matrix
    
    return js_matrix

class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []
        self.sent1 = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])

            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()], dtype=object)
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1], dtype=object)[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2], dtype=object)[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2
            self.sent1 += sent1

        

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def test(self, params, batcher):

        batch = [['work', 'into', 'it', 'slowly', '.']]
        enc, attn, last_records, _ = batcher(params, batch)
        result_matrix = calculate_js_matrix(attn)
        print(result_matrix)

        # batch1 = [['you', 'should', 'never', 'do', 'it', '.']]  #sentence1
        # batch2 = [['how', 'do', 'you', 'do', 'that', '?']]    #sentence2

        # ================================= batch2: ['no', ',', 'it', 'makes', 'no', 'difference', '.']
        # ================================= match2: ['i', 'was', 'in', 'a', 'similar', 'situation', '.']
        
        # batch1 = [['what', 'are', 'these', 'bugs', 'and', 'how', 'do', 'i', 'get', 'rid', 'of', 'them', '?']]#sentence3
        # batch2 = [['you', 'have', 'to', 'decide', 'what', 'you', 'want', 'to', 'get', 'out', 'of', 'this', '.']]#sentence4
        # batch1 = [['a', 'couple', 'things', 'to', 'consider', ':']] 
        # batch2 = [['take', 'a', 'look', 'at', 'these', ':']]
        # batch1 = [['you', 'are', 'right', 'on', 'the', 'mark', '.']]
        # batch2 = [['what', 'is', 'this', 'vocal', 'technique', 'called', '?']]
        # batch1 = [['how', 'do', 'i', 'patch', 'a', 'gap', 'between', 'my', 'bathtub', 'and', 'wall', '?']]
        # batch2 = [['what', 'goes', 'in', 'a', 'student', 'success', 'statement', 'for', 'a', 'faculty', 'application', '?']]

        batch1 = [['how', 'can', 'i', 'add', 'a', 'new', 'light', 'fixture', 'off', 'of', 'a', 'ceiling', 'fan', 'wired', 'to', 'two', 'switches', '?']]
        batch2 = [['the', 'order', 'in', 'which', 'the', 'terms', 'appear', 'in', 'the', 'document', 'is', 'lost', 'in', 'the', 'vector', 'space', 'representation', '.']]
        enc1, attn1, last_records1 = batcher(params, batch1) 
        enc2, attn2, last_records2 = batcher(params, batch2)

        # print(f"================================= batch1: {batch1[0]}")
        # print(f"================================= match1: {self.sent1[(last_records1[0][0] - 27) // 28]}")

        # print(f"================================= batch2: {batch2[0]}")
        # print(f"================================= match2: {self.sent1[(last_records2[0][0] - 27) // 28]}")

        dir = f"/home/sdh/MetaEOL/MetaEOL/figures"
        # head=20
        size=28

        cmap = sns.color_palette("coolwarm", as_cmap=True)
        cmap.set_bad(color="gray") 
        for layer in range(28):
            dir1 = f"{dir}/sentence3/layer_{layer}"
            if not os.path.exists(dir1):
                os.makedirs(dir1)
        
        for layer in range(28):
            dir2 = f"{dir}/sentence4/layer_{layer}"
            if not os.path.exists(dir2):
                os.makedirs(dir2)

        for layer in range(28): # 28 layers
            apm0 = attn1[layer].numpy() # 24 heads 
            apm1 = attn2[layer].numpy()

            for head in range(24): # 24 heads
           
                mask=np.triu(np.ones_like(apm0[0][head][-size:, -size:,], dtype=bool), k=1)
                sns.heatmap(apm0[0][head][-size:, -size:,], cmap=cmap, vmin=-0.1,linewidths=0, rasterized=True, square=True, vmax= 1, mask=mask)
                plt.title(f"Sentence 3 Layer {layer} Head {head}", fontsize=15)
                plt.xticks(ticks=np.arange(0, size, 2), labels=np.arange(0, size, 2))
                plt.yticks(ticks=np.arange(0, size, 2), labels=np.arange(0, size, 2), rotation=360)
                plt.show()
                plt.savefig(f"{dir}/sentence3/layer_{layer}/Sentence_3_layer_{layer}_head_{head}.pdf", format="pdf", bbox_inches="tight")
                plt.clf()

                mask=np.triu(np.ones_like(apm1[0][head][-size:, -size:,], dtype=bool), k=1)
                sns.heatmap(apm1[0][head][-size:, -size:,], cmap=cmap, vmin=-0.1,linewidths=0, rasterized=True, square=True, vmax= 1, mask=mask)
                plt.title(f"Sentence 4 Layer {layer} Head {head}", fontsize=15)
                plt.xticks(ticks=np.arange(0, size, 2), labels=np.arange(0, size, 2))
                plt.yticks(ticks=np.arange(0, size, 2), labels=np.arange(0, size, 2), rotation=360)
                plt.show()
                plt.savefig(f"{dir}/sentence4/layer_{layer}/Sentence_4_layer_{layer}_head_{head}.pdf", format="pdf", bbox_inches="tight")
                plt.clf()


           


        

                print("ok")
            ################################################



        print("ok")

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]

            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1, attn1, last_records1, _ = batcher(params, batch1)
                    print(f"================================= batch1: {batch1[0]}")
                    if last_records1 is not None and len(last_records1) != 0:
                        print(f"================================= match1: {self.sent1[(last_records1[0][0] - 27) // 28]}")
                    else:
                        print(f"================================= no match1")
                    
                    enc2, attn2, last_records2, _ = batcher(params, batch2)
                    print(f"================================= batch2: {batch2[0]}")
                    if last_records2 is not None and len(last_records2) != 0:
                        print(f"================================= match2: {self.sent1[(last_records2[0][0] - 27) // 28]}")
                    else:
                        print(f"================================= no match2")

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
                    print(f"================================= y^hat: {sys_score} y: {gs_scores[ii]}")
            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0]))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results
    
   
    def collect_apms_hs(self, params, batcher):
        sample_num = len(self.sent1)
        # random_sample = random.sample(self.samples, sample_num)
        # print(random_sample)  
        for ii in range(0, sample_num):
            batch = [self.sent1[ii]]
            embedding = batcher(params, batch)
        logging.debug('Collect %.2f Hidden_states Att Masks and APMs Success!' % (sample_num))
        


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        #train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        #dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        #test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        #self.datasets = ['train', 'dev', 'test']
        #self.data = {'train': train, 'dev': dev, 'test': test}
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['test']
        self.data = {'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkEvalDev(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        #train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        #dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        #test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        #self.datasets = ['train', 'dev', 'test']
        #self.data = {'train': train, 'dev': dev, 'test': test}
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        self.datasets = ['dev']
        self.data = {'dev': dev}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkFinetune(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        #train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        #dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        #test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        #self.datasets = ['train', 'dev', 'test']
        #self.data = {'train': train, 'dev': dev, 'test': test}
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['test']
        self.data = {'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
