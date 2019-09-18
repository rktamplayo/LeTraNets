import numpy as np
import tensorflow as tf

fc_layer = tf.contrib.layers.fully_connected
xav_init = tf.contrib.layers.xavier_initializer

class LeRNets(object):

    def __init__(self,
                 vocab_size,
                 sentence_len,
                 segment_len,
                 sequence_len,
                 num_classes,
                 joint_train=False,
                 pred_reg=False,
                 embed_size=300):
        
        self.embed_size = embed_size
        self.sequence_len = sequence_len
        sequence_len += 8
        segment_len += 8

        self.full_inputs = tf.placeholder(tf.int32, shape=[None, sequence_len])
        self.seg_inputs = tf.placeholder(tf.int32, shape=[None, sentence_len, segment_len])
        self.sen_length = tf.placeholder(tf.float32, shape=[None, sentence_len])
        self.sentiment = tf.placeholder(tf.int32, shape=[None, num_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.embedding = tf.get_variable("embedding", [vocab_size, embed_size], initializer=xav_init())
        #self.s_embedding = tf.get_variable("s_embedding", [vocab_size, embed_size], initializer=xav_init())
        
        self.f2f_vector = self.cnn(self.full_inputs, self.embedding, sequence_len, "full_cnn")

        seg_inputs = tf.transpose(self.seg_inputs, perm=[1,0,2])
        self.seg_inp = seg_inputs
        
        self.s2s_vector = tf.map_fn(lambda x: self.cnn(x, self.embedding, segment_len, "seg_cnn"), seg_inputs, dtype=tf.float32)
        self.s2f_vector = tf.map_fn(lambda x: self.cnn(x, self.embedding, segment_len, "full_cnn"), seg_inputs, dtype=tf.float32)
        self.s2s_vector = tf.transpose(self.s2s_vector, perm=[1,0,2])
        self.s2f_vector = tf.transpose(self.s2f_vector, perm=[1,0,2])
        
        Wf = tf.get_variable("Wf", [embed_size, num_classes], initializer=xav_init())
        bf = tf.Variable(tf.constant(0.0, shape=[num_classes]))
        self.f_scores = tf.nn.xw_plus_b(self.f2f_vector, Wf, bf)
        
        Ws = tf.get_variable("Ws", [embed_size, num_classes], initializer=xav_init())
        bs = tf.Variable(tf.constant(0.0, shape=[num_classes]))
        self.f2s_vector, self.attention = self.attend(self.s2s_vector, self.sen_length)
        #self.f2s_vector = tf.reduce_mean(self.s2s_vector, 1)
        self.s_scores = tf.nn.xw_plus_b(self.f2s_vector, Ws, bs)
        
        with tf.variable_scope('whole'):
            Wa = tf.get_variable("Wa", [embed_size*2, num_classes], initializer=xav_init())
            ba = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="ba")
        self.a_scores = tf.nn.xw_plus_b(tf.concat([self.f2f_vector, self.f2s_vector], -1), Wa, ba)
        
        if pred_reg:
            s2s_scores = tf.nn.softmax(tf.nn.xw_plus_b(tf.reshape(self.s2s_vector, [-1, 300]), Ws, bs))
            s2f_scores = tf.nn.softmax(tf.nn.xw_plus_b(tf.reshape(self.s2f_vector, [-1, 300]), Wf, bf))

            seg_reg = tf.reduce_sum(s2s_scores * tf.log(s2s_scores/s2f_scores), -1)
            #seg_reg = tf.nn.softmax_cross_entropy_with_logits(logits=s2f_scores, labels=s2s_scores)
            #seg_reg = tf.losses.absolute_difference(labels=s2f_scores, predictions=s2s_scores)
            seg_reg = tf.reduce_mean(seg_reg)
        else:
            seg_reg = 0
        
        s_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_scores, labels=self.sentiment)
        self.s_loss = tf.reduce_mean(s_losses)
        self.s_updates = self.apply_grads(self.s_loss)
        
        f_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.f_scores, labels=self.sentiment)
        self.f_loss = tf.reduce_mean(f_losses) + pred_reg * seg_reg
        self.f_updates = self.apply_grads(self.f_loss, "full_cnn")
            
        if joint_train:
            a_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.a_scores, labels=self.sentiment)
            self.a_loss = tf.reduce_mean(a_losses)
            self.a_updates = self.apply_grads(self.a_loss, "whole")
        else:
            self.a_loss = 0
        
        self.loss = self.f_loss + self.s_loss + self.a_loss
        self.updates = self.apply_grads(self.loss)
        
        self.scores = tf.nn.softmax(self.f_scores) + tf.nn.softmax(self.s_scores)
        if joint_train:
            self.scores += tf.nn.softmax(self.a_scores)
            
        self.actual = tf.argmax(self.sentiment, 1)
        
        self.f_count, self.f_se = self.evaluate(self.f_scores)
        self.s_count, self.s_se = self.evaluate(self.s_scores)
        self.a_count, self.a_se = self.evaluate(self.a_scores)
        self.count, self.se = self.evaluate(self.scores)
        
        self.saver = tf.train.Saver(tf.global_variables())
    
    def evaluate(self, scores):
        predicted = tf.argmax(scores, 1)
        correct = tf.cast(tf.equal(predicted, self.actual), 'float')
        count = tf.reduce_sum(correct)
        se = tf.reduce_sum(tf.square(predicted-self.actual))
        
        return count, se
    
    def apply_grads(self, loss, scope=None):
        optimizer = tf.train.AdadeltaOptimizer(1.0, 0.95, 1e-6)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope) 
        grads_and_vars = optimizer.compute_gradients(loss, var_list=vars)
        capped = []
        for gv in grads_and_vars:
            try:
                clipped = (tf.clip_by_norm(gv[0], clip_norm=3, axes=[0]), gv[1])
            except:
                continue
            capped.append(clipped)
        updates = optimizer.apply_gradients(capped)
        return updates
    
    def cnn(self, inputs, embed, sequence_len, name):
        embedding = tf.nn.embedding_lookup(embed, inputs)
        embedding = tf.expand_dims(embedding, -1)
    
        filter_sizes = [3,4,5]
        num_filters = [100,100,100]
        
        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope(name + "-conv-%s" % filter_size, reuse=tf.AUTO_REUSE):
                filter_shape = [filter_size, self.embed_size, 1, num_filters[i]]
                W = tf.get_variable("W", filter_shape, initializer=xav_init())
                b = tf.get_variable("b", [num_filters[i]], initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(
                    embedding,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1,sequence_len-filter_size+1,1,1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="pool")
                outputs.append(pooled)
        
        outputs = tf.concat(outputs, axis=3)
        outputs = tf.reshape(outputs, [-1, self.embed_size])
        outputs = tf.nn.dropout(outputs, self.keep_prob)
        
        return outputs
    
    def attend(self, vector, length):
        X = tf.get_variable("X", shape=[self.embed_size, self.embed_size], initializer=xav_init())
        b = tf.Variable(tf.zeros([self.embed_size]))
        z = tf.get_variable("z", shape=[self.embed_size], initializer=xav_init())
        
        sem = tf.tensordot(vector, X, 1)
        sem.set_shape(vector.shape)
        
        weights = tf.nn.tanh(sem + b)
        
        weights = tf.tensordot(weights, z, 1)
        weights.set_shape(vector.shape[:-1])
        
        weights = tf.nn.softmax(weights+length)
        weights = tf.expand_dims(weights, -1)
        
        attended = tf.multiply(vector, weights)
        attended = tf.reduce_sum(attended, 1)
        
        return attended, weights
    
    def step(self,
             session,
             full_inputs,
             seg_inputs,
             sen_length,
             sentiment,
             max_sen_len,
             max_seg_len,
             max_seq_len,
             training="all"):
        
        full_inputs = self.full_pad(full_inputs, max_seq_len)
        seg_inputs = self.seg_pad(seg_inputs, max_sen_len, max_seg_len)
        
        full_inputs = np.array(full_inputs)
        seg_inputs = np.array(seg_inputs)
        
        input_feed = {}
        input_feed[self.full_inputs] = full_inputs
        input_feed[self.seg_inputs] = seg_inputs
        input_feed[self.sen_length] = sen_length
        input_feed[self.sentiment] = sentiment
        
        if training == "full":
            input_feed[self.keep_prob] = 0.5
            output_feed = [self.f_loss, self.f_updates]
        elif training == "seg":
            input_feed[self.keep_prob] = 0.5
            output_feed = [self.s_loss, self.s_updates]
        elif training == "all":
            input_feed[self.keep_prob] = 0.5
            output_feed = [self.a_loss, self.a_updates]
        elif training == "whole":
            input_feed[self.keep_prob] = 0.5
            output_feed = [self.loss, self.updates]
        elif training == 'test_full':
            input_feed[self.keep_prob] = 1.0
            output_feed = [self.f_count, self.f_se, self.attention]
        elif training == 'test_seg':
            input_feed[self.keep_prob] = 1.0
            output_feed = [self.s_count, self.s_se, self.attention]
        elif training == 'test_all':
            input_feed[self.keep_prob] = 1.0
            output_feed = [self.a_count, self.a_se, self.attention]
        else:
            input_feed[self.keep_prob] = 1.0
            output_feed = [self.count, self.se, self.attention]
        
        outputs = session.run(output_feed, input_feed)
        
        return outputs
    
    def full_pad(self,
                 inputs,
                 max_len):
        new_inp = []
        for input in inputs:
            input = list(input)
            input += [0] * (max_len - len(input))
            input = [0] * 4 + input + [0] * 4
            new_inp.append(input[:max_len+8])
        return new_inp
    
    def seg_pad(self,
                inputs,
                max_sen_len,
                max_seg_len):
        new_inp = []
        for input in inputs:
            new_seg = []
            for seg in input:
                seg = list(seg)
                seg += [0] * (max_seg_len - len(seg))
                seg = [0] * 4 + seg + [0] * 4
                seg = seg[:max_seg_len+8]
                new_seg.append(seg)
            temp = [0] * (max_seg_len+8)
            new_seg += [temp] * (max_sen_len - len(new_seg))
            new_seg = new_seg[:max_sen_len]
            new_inp.append(new_seg)
        return new_inp