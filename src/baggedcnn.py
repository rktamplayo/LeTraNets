import numpy as np
import tensorflow as tf

fc_layer = tf.contrib.layers.fully_connected
xav_init = tf.contrib.layers.xavier_initializer

class BaggedCNN(object):

    def __init__(self,
                 vocab_size,
                 sentence_len,
                 segment_len,
                 num_classes,
                 embed_size=300):
        
        self.embed_size = embed_size
        segment_len += 8
        
        self.inputs = tf.placeholder(tf.int32, shape=[None, sentence_len, segment_len])
        self.sen_length = tf.placeholder(tf.float32, shape=[None, sentence_len])
        self.sentiment = tf.placeholder(tf.int32, shape=[None, num_classes])
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.embedding = tf.get_variable("embedding", [vocab_size, embed_size], initializer=xav_init())
        
        inputs = tf.transpose(self.inputs, perm=[1,0,2])
        self.segments = tf.map_fn(lambda x: self.cnn(x, self.embedding, segment_len, "cnn"), inputs, dtype=tf.float32)
        self.segments = tf.transpose(self.segments, perm=[1,0,2])
        self.document, _ = self.attend(self.segments, self.sen_length)
        
        W = tf.get_variable("W", [embed_size, num_classes], initializer=xav_init())
        b = tf.Variable(tf.constant(0.0, shape=[num_classes]))
        self.scores = tf.nn.xw_plus_b(self.document, W, b)
        
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.sentiment)
        self.loss = tf.reduce_mean(losses)
        self.updates = self.apply_grads(self.loss)
        
        self.predicted = tf.argmax(self.scores, 1)
        self.actual = tf.argmax(self.sentiment, 1)
        
        correct = tf.cast(tf.equal(self.predicted, self.actual), 'float')
        self.count = tf.reduce_sum(correct)
        self.se = tf.reduce_sum(tf.square(self.predicted-self.actual))
        
        self.saver = tf.train.Saver(tf.global_variables())
    
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
             seg_inputs,
             sen_length,
             sentiment,
             max_sen_len,
             max_seg_len,
             training=True):
        
        seg_inputs = self.seg_pad(seg_inputs, max_sen_len, max_seg_len)
        seg_inputs = np.array(seg_inputs)
        
        input_feed = {}
        input_feed[self.inputs] = seg_inputs
        input_feed[self.sen_length] = sen_length
        input_feed[self.sentiment] = sentiment
        
        if training:
            input_feed[self.keep_prob] = 0.5
            output_feed = [self.loss, self.updates]
        else:
            input_feed[self.keep_prob] = 1.0
            output_feed = [self.count, self.se]
        
        outputs = session.run(output_feed, input_feed)
        
        return outputs
    
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