import tensorflow as tf
import numpy as np
import lernets
import utils
import sys

def fl(float):
    return "%0.3f" % float

data1_type = sys.argv[1] # movies, naver, restaurants
train_type = sys.argv[2]
data2_type = sys.argv[3] # movies, naver, restaurants
test_type = sys.argv[4]
fg_type = int(sys.argv[5])
type = sys.argv[6]
joint_train, pred_reg, grad_change, step_pretrain = [float(x) for x in sys.argv[7].split(',')]
print(bool(joint_train))

# FILE LOCATION
data1_dir = 'data/' + data1_type
train_file = data1_dir + '/' + train_type + '_train.dat'
data2_dir = 'data/' + data2_type
dev_file = data2_dir + '/' + test_type + '_test.dat'
vec_file = 'glove.42B.300d.txt'
pickle_file = data2_dir + '/lernets_' + type + '.p'

# PARAMETERS
epoch = 10
x_index = 0
if fg_type == 0:
    y_index = 2
else:
    y_index = 1
if fg_type == 0:
    num_classes = 2
elif data2_type == 'restaurants':
    num_classes = 5
else:
    num_classes = 10
batch_size = 32
st = 100

import os
import time
import pickle
if os.path.isfile(pickle_file):
    x_f_train, x_s_train, x_len_train, y_train, \
        x_f_dev, x_s_dev, x_len_dev, y_dev, \
        x_dict, word_vectors, \
        max_sen_len, max_seg_len, max_seq_len = pickle.load(open(pickle_file, 'rb'))
else:
    x_dict = utils.get_flat_dict(train_file, x_index)
    word_vectors = utils.get_vectors(x_dict, vec_file)
    x_f_train, y_train, len_f_train = utils.get_flat_data(train_file, x_index, y_index, x_dict, num_classes)
    x_s_train, _, x_len_train, max_train_sen, len_s_train = utils.get_hier_data(train_file, x_index, y_index, x_dict, num_classes)
    x_f_dev, y_dev, len_f_dev = utils.get_flat_data(dev_file, x_index, y_index, x_dict, num_classes)
    x_s_dev, _, x_len_dev, max_dev_sen, len_s_dev = utils.get_hier_data(dev_file, x_index, y_index, x_dict, num_classes, max_train_sen)
    x_s_train, _, x_len_train, max_train_sen, len_s_train = utils.get_hier_data(train_file, x_index, y_index, x_dict, num_classes, max_dev_sen)
    max_sen_len = max(max_train_sen, max_dev_sen)
    max_seg_len = max(len_s_train, len_s_dev)
    max_seq_len = max(len_f_train, len_f_dev)
    pickle.dump([x_f_train, x_s_train, x_len_train, y_train, \
                 x_f_dev, x_s_dev, x_len_dev, y_dev, \
                 x_dict, word_vectors, \
                 max_sen_len, max_seg_len, max_seq_len], \
                 open(pickle_file, 'wb'), protocol=4)

vocab_size = len(x_dict)

model = lernets.LeRNets(vocab_size, max_sen_len, max_seg_len, max_seq_len, num_classes, joint_train, pred_reg)
trainings = ['whole']
if step_pretrain:
    trs = ['seg', 'full']
    if joint_train:
        trs.append('all')
    trainings = trs + trainings

print(max_sen_len)

sess = tf.Session()
tf.set_random_seed(1234)
np.random.seed(1234)

sess.run(tf.global_variables_initializer())
sess.run(model.embedding.assign(word_vectors))
#sess.run(model.s_embedding.assign(word_vectors))

import time
import math
step = 0
cur_time = time.time()
best_acc = 0
best_accs = {}
best_rmse = 0

with sess.as_default():
    for training in trainings:
        best_accs[training] = 0
        print('training on', training)
        for i in range(epoch):
            shuffle_indices = np.random.permutation(np.arange(len(y_train)))
            x_f_train_shuffle = x_f_train[shuffle_indices]
            x_s_train_shuffle = x_s_train[shuffle_indices]
            x_len_train_shuffle = x_len_train[shuffle_indices]
            y_train_shuffle = y_train[shuffle_indices]
            
            train_loss = []
            for j in range(0, len(x_f_train_shuffle), batch_size):
                x_f_batch = x_f_train_shuffle[j:j+batch_size]
                x_s_batch = x_s_train_shuffle[j:j+batch_size]
                x_len_batch = x_len_train_shuffle[j:j+batch_size]
                y_batch = y_train_shuffle[j:j+batch_size]
                
                loss, _ = model.step(sess, x_f_batch, x_s_batch, x_len_batch, y_batch, max_sen_len, max_seg_len, max_seq_len, training=training)
                train_loss.append(loss)
                
                step += batch_size
                if step > st:
                    total_se = 0
                    total_count = 0
                    for k in range(0, len(y_dev), batch_size):
                        x_f_batch = x_f_dev[k:k+batch_size]
                        x_s_batch = x_s_dev[k:k+batch_size]
                        x_len_batch = x_len_dev[k:k+batch_size]
                        y_batch = y_dev[k:k+batch_size]
                        
                        count, se, attention = model.step(sess, x_f_batch, x_s_batch, x_len_batch, y_batch, max_sen_len, max_seg_len, max_seq_len, training="test_" + training)
                        total_count += count
                        total_se += se
                    acc = total_count / len(x_f_dev)
                    rmse = math.sqrt(total_se / len(x_f_dev))
                    if acc >= best_acc:
                        best_acc = acc
                        best_rmse = rmse
                    if acc >= best_accs[training]:
                        best_accs[training] = acc
                        model.saver.save(sess, data2_dir + '/lernets_model_' + type)
                    
                    print('epoch', i, 'instance', j, 'train loss', fl(np.mean(train_loss)), 'test accs', fl(acc), 'best accs', fl(best_acc), 'best rmses', fl(best_rmse), 'time', time.time()-cur_time)
                    
                    step -= st
                    cur_time = time.time()
                    train_loss = []
        tf.reset_default_graph()
        model.saver.restore(sess, data2_dir + '/lernets_model_' + type)