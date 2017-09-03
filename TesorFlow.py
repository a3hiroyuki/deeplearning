'''
Created on 2017/09/04

@author: Abe
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

class TesorFlow:
    
    def __init__(self, tr_features, tr_labels, ts_features, ts_labels, label_num):
        self.tr_features = tr_features
        self.tr_labels = tr_labels
        self.ts_features = ts_features
        self.ts_labels = ts_labels
        self.n_classes = label_num
        self.init_params()
        self.create_neural_network()
        
    def init_params(self):
        self.training_epochs = 50
        self.n_dim = self.tr_features.shape[1]
        #self.n_classes = 2
        #self.n_hidden_units_one = 280 
        #self.n_hidden_units_two = 300
        self.n_hidden_units_one = 100 
        self.n_hidden_units_two = 100
        self.sd = 1 / np.sqrt(self.n_dim)
        self.learning_rate = 0.005
        
    def create_neural_network(self):  
        self.X = tf.placeholder(tf.float32,[None,self.n_dim])
        self.Y = tf.placeholder(tf.float32,[None,self.n_classes])
      
        W_1 = tf.Variable(tf.random_normal([self.n_dim,self.n_hidden_units_one], mean = 0, stddev=self.sd))
        b_1 = tf.Variable(tf.random_normal([self.n_hidden_units_one], mean = 0, stddev=self.sd))
        h_1 = tf.nn.tanh(tf.matmul(self.X,W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([self.n_hidden_units_one,self.n_hidden_units_two], mean = 0, stddev=self.sd))
        b_2 = tf.Variable(tf.random_normal([self.n_hidden_units_two], mean = 0, stddev=self.sd))
        h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

        W = tf.Variable(tf.random_normal([self.n_hidden_units_two,self.n_classes], mean = 0, stddev=self.sd))
        b = tf.Variable(tf.random_normal([self.n_classes], mean = 0, stddev=self.sd))
        self.y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)
        
        self.cost_function = -tf.reduce_sum(self.Y * tf.log(self.y_))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_function)

        correct_prediction = tf.equal(tf.argmax(self.y_,1), tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
    def execute(self):
        cost_history = None
        y_true, y_pred = None, None
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            cost_history = self.learn(sess)
            y_true, y_pred = self.test(sess)
        fig = plt.figure(figsize=(10,8))
        plt.plot(cost_history)
        plt.axis([0,self.training_epochs,0,np.max(cost_history)])
        plt.show()
        print(y_true, y_pred)
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
        print ("F-Score:", round(f,3))
        
    def learn(self, sess):
        cost_history = np.empty(shape=[1], dtype=float)
        for epoch in range(self.training_epochs):            
            _, cost = sess.run([self.optimizer, self.cost_function], feed_dict={self.X:self.tr_features, self.Y: self.tr_labels})
            cost_history = np.append(cost_history, cost)
        return cost_history

    def test(self, sess):
        y_pred = sess.run(tf.argmax(self.y_,1),feed_dict={self.X: self.ts_features})
        y_true = sess.run(tf.argmax(self.ts_labels,1))
        print("Test accuracy: ", sess.run(self.accuracy, feed_dict={self.X: self.ts_features, self.Y: self.ts_labels}))
        return y_pred, y_true
        