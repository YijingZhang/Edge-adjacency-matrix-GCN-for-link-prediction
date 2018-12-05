import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self):
        pass

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        return var

    def gc(self, x, L, Mout, activate=tf.nn.relu):
        '''
        Graph convolution layer with Mout features
        '''
        # L.dtype = x.dtype
        N, Min = x.get_shape()
        N, Min = int(N), int(Min)
        W = self._weight_variable([Min, Mout])
        b = self._bias_variable([Mout])
        x = tf.matmul(L, x)  # M x N*Min
        if not activate:
            return tf.matmul(x, W) + b
        x = activate(tf.matmul(x, W) + b)  # N x M x Mout
        return x

    def gcn(self, x, L, GcM):
        """
        Filtering with GCN interpolation
        Implementation: numpy.
        """
        N, Fin = x.shape
        N, Fin = int(N), int(Fin)
        # Graph convolution
        for i, M in enumerate(GcM):
            with tf.variable_scope('gc{}'.format(i + 1)):
                if i == len(GcM) - 1:
                    x = self.gc(x, L, M, activate=None)
                else:
                    x = self.gc(x, L, M)
        return x

    def train(self, L, GcM, labelled_id):
        lr = 0.01
        node_num,_ = np.shape(L)
        class_num = GcM[-1]
        labelled_num = len(labelled_id)
        x = tf.placeholder(tf.float32, [node_num, node_num])
        y = tf.placeholder(tf.float32, [labelled_num, class_num])
        logits = self.gcn(x, L, GcM)
        y_predict = tf.nn.softmax(logits)
        loss_op = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.gather(logits, labelled_id, axis=0),
                                                       labels=y))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss_op)
        pred_labels = tf.argmax(y_predict, axis=1)
        return {'train_op': train_op,
                'loss_op': loss_op,
                'y_predict': y_predict,
                'x': x,
                'y': y,
                'pred_labels': pred_labels
                }

