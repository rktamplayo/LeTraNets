import os
import html
import numpy as np
import pandas as pd
import tensorflow as tf

def get_hier_dict(data_dir, x_index):
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    count = {}
    for line in f:
        line = line.split('\t\t')
        x_par = line[x_index].split('<sssss>')
        for x_sen in x_par:
            x_inp = x_sen.strip().split()
            for term in x_inp:
                if term not in count:
                    count[term] = 0
                count[term] += 1
    f.close()
    
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    x_dict = {}
    x_dict['<PAD>'] = len(x_dict)
    for line in f:
        line = line.split('\t\t')
        x_par = line[x_index].split('<sssss>')
        for x_sen in x_par:
            x_inp = x_sen.strip().split()
            for term in x_inp:
                if term not in x_dict:
                    x_dict[term] = len(x_dict)
    f.close()
    
    return x_dict

def get_flat_dict(data_dir, x_index):
    count = {}
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split('\t\t')
        x_inp = line[x_index].strip().split()
        for term in x_inp:
            if term not in count:
                count[term] = 0
            count[term] += 1
    f.close()

    x_dict = {}
    x_dict['<PAD>'] = len(x_dict)
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split('\t\t')
        x_inp = line[x_index].strip().split()
        for term in x_inp:
            if term not in x_dict:
                x_dict[term] = len(x_dict)
    f.close()

    return x_dict

def get_vectors(x_dict, vec_file, emb_size=300):
    x_vectors = np.random.uniform(-0.1, 0.1, (len(x_dict), emb_size))
    
    if vec_file != None:
        f = open(vec_file, 'r', encoding='utf-8', errors='ignore')
        for line in f:
            line = line.split()
            if line[0] in x_dict:
                x_vectors[x_dict[line[0]]] = np.array([float(x) for x in line[-emb_size:]])
        f.close()
    
    return x_vectors

def get_vectors2(x_dict, vec_file1, vec_file2, emb_size=300):
    x_vectors1 = np.random.uniform(-0.1, 0.1, (len(x_dict), 300))
    x_vectors2 = np.random.uniform(-0.1, 0.1, (len(x_dict), 50))
    
    f = open(vec_file1, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split()
        if line[0] in x_dict:
            x_vectors1[x_dict[line[0]]] = np.array([float(x) for x in line[-300:]])
    f.close()

    f = open(vec_file2, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split()
        if line[0] in x_dict:
            x_vectors2[x_dict[line[0]]] = np.array([float(x) for x in line[-50:]])
    f.close()
    
    return x_vectors1, x_vectors2

def get_hier_data(data_dir, x_index, y_index, x_dict, max_sent, prev_max_sen=0, slice=100):
    x_dat = []
    y_dat = []
    max_sen_len = 0
    max_seq_len = 0
    
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split('\t\t')
        
        x = []
        x_par = line[x_index].split()
        for i in range(0, len(x_par), slice):
            x_indices = []
            for term in x_par[i:i+slice]:
                if term not in x_dict:
                    continue
                if term in ['.', '...', '!', '?']:
                    x_indices.append(x_dict[term])
                    max_seq_len = max(max_seq_len, len(x_indices))
                    x.append(x_indices)
                    x_indices = []
                else:
                    x_indices.append(x_dict[term])
            if len(x_indices) != 0:
                x.append(x_indices)
            
        max_sen_len = max(max_sen_len, len(x))
        
        y = np.zeros(max_sent)
        if max_sent == 2:
            y[int(line[y_index])] = 1
        else:
            y[int(line[y_index])-1] = 1
        
        x_dat.append(x)
        y_dat.append(y)
    f.close()
    
    max_sen_len = max(prev_max_sen, max_sen_len)
    x_len_dat = []
    for x in x_dat:
        len_vec = np.concatenate((np.zeros(len(x)), np.full(max_sen_len-len(x), -1000000000.0)))
        x_len_dat.append(len_vec)
    
    x_dat = np.array(x_dat)
    y_dat = np.array(y_dat)
    x_len_dat = np.array(x_len_dat)
    
    return x_dat, y_dat, x_len_dat, max_sen_len, max_seq_len

def get_flat_data(data_dir, x_index, y_index, x_dict, max_sent):
    x_dat = []
    y_dat = []
    max_len = 0
    
    f = open(data_dir, 'r', encoding='utf-8', errors='ignore')
    for line in f:
        line = line.split('\t\t')
        
        x = []
        x_inp = line[x_index].strip().split()
        for term in x_inp:
            if term not in x_dict:
                continue
            else:
                x.append(x_dict[term])
        
        max_len = max(max_len, len(x))
        
        y = np.zeros(max_sent)
        if max_sent == 2:
            y[int(line[y_index])] = 1
        else:
            y[int(line[y_index])-1] = 1
        
        x_dat.append(x)
        y_dat.append(y)
    f.close()
    
    x_dat = np.array(x_dat)
    y_dat = np.array(y_dat)
    
    return x_dat, y_dat, max_len
