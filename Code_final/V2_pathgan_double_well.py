#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import genfromtxt
from aux_functions import xavier_init,plot_withDec,mix_rbf_mmd2,update_weights_gen,update_weights_dis

import csv
import sys

epochs = int(sys.argv[1])  # input from command line
iteration = int(sys.argv[2])  # input from command line
beta = float(sys.argv[3])
gamma = float(sys.argv[4])

print('beta value:', beta)
print('gamma value:', gamma)

mb_size = 10000
D_h1_dim=10
D_h2_dim=10
#D_h3_dim=4 #layers and nodes of discriminator
Z_dim = 2
G_h1_dim=10
G_h2_dim=10 #layers and nodes of generator
#G_h3_dim=6

X_dim = 2

dt = 0.01

X = tf.placeholder(tf.float32, shape=[None, X_dim])
Xpre = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([2*X_dim, D_h1_dim]))
#D_W1 = tf.Variable(xavier_init([X_dim, D_h1_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[D_h1_dim]))

D_W2 = tf.Variable(xavier_init([D_h1_dim, D_h2_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[D_h2_dim]))

D_W3 = tf.Variable(xavier_init([D_h2_dim, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))
'''
D_W3 = tf.Variable(xavier_init([D_h2_dim, D_h3_dim]))
D_b3 = tf.Variable(tf.zeros(shape=[D_h3_dim]))

D_W4 = tf.Variable(xavier_init([D_h3_dim, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))
'''    
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
#theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]

Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([X_dim, G_h1_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[G_h1_dim]))

G_W2 = tf.Variable(xavier_init([G_h1_dim, G_h2_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[G_h2_dim]))

G_W3 = tf.Variable(xavier_init([G_h2_dim, X_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[X_dim]))
'''
G_W3 = tf.Variable(xavier_init([G_h2_dim, G_h3_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[G_h3_dim]))

G_W4 = tf.Variable(xavier_init([G_h3_dim, X_dim]))
G_b4 = tf.Variable(tf.zeros(shape=[X_dim]))#
'''
sigma = tf.Variable(tf.ones(shape=[Z_dim]))
#sigma = 0.15

#theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4, sigma]
theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, sigma]

def drift(x):
    G_h1 = tf.nn.elu(tf.matmul(x, G_W1) + G_b1) 
    G_h2 = tf.nn.elu(tf.matmul(G_h1, G_W2) + G_b2)
    #G_h3 = tf.nn.tanh(tf.matmul(G_h2, G_W3) + G_b3)
    
    #G_log_prob = tf.matmul(G_h3, G_W4) + G_b4
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
    return G_log_prob
'''
def drift(x):
    G_log_prob = tf.matmul(x, G_W1) + tf.matmul(x**2, G_W2) + tf.matmul(x**3, G_W3)
    return G_log_prob
'''
def generator(x_,z):
    
    G_log_prob = drift(x_)
   
    return x_ + dt*G_log_prob + sigma*tf.sqrt(dt)*z
   

##G_W1 = tf.Variable(xavier_init([X_dim, X_dim]))
##G_W2 = tf.Variable(xavier_init([X_dim, X_dim]))
##G_W3 = tf.Variable(xavier_init([X_dim, X_dim]))
##G_W4 = tf.Variable(xavier_init([Z_dim, X_dim]))
#
#G_W1 = tf.Variable(tf.zeros([X_dim, X_dim]))
#G_W2 = tf.Variable(tf.zeros([X_dim, X_dim]))
#G_W3 = tf.Variable(tf.zeros([X_dim, X_dim]))
#G_W4 = tf.Variable(tf.zeros([Z_dim, X_dim]))
#
#def generator(x_,z):
#    G_log_prob = x_ + dt*tf.matmul(x_, G_W1) + dt*tf.matmul(x_**2, G_W2) + dt*tf.matmul(x_**3, G_W3) + tf.sqrt(dt)*tf.matmul(z, G_W4)
##    G_log_prob = x_ + 0.01*tf.matmul(x_, G_W1) + 0.01*tf.matmul(x_**3, G_W3) + 0.1*z# tf.matmul(z, G_W4)
#    return G_log_prob

#theta_G = [G_W1, G_W2, G_W3, G_W4]
##theta_G = [G_W1, G_W3]#, G_W4]

def discriminator(x_, x):
    D_h1 = tf.nn.relu(tf.matmul(tf.concat([x_,x],1), D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    #D_h3 = tf.nn.tanh(tf.matmul(D_h2, D_W3) + D_b3)
    
    #out = tf.matmul(D_h3, D_W4) + D_b4
    out = tf.matmul(D_h2, D_W3) + D_b3
#    out = 100.0 * tf.nn.tanh((tf.matmul(D_h2, D_W3) + D_b3) / 100.0) # apply boundedness condition
    return out

#def discriminator(x):
#    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
#    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
#    out = tf.matmul(D_h2, D_W3) + D_b3 
##    out = 100.0 * tf.nn.tanh((tf.matmul(D_h2, D_W3) + D_b3) / 100.0) # apply boundedness condition
#    return out

def sample_Z(m, n):
    return np.random.normal(0., 1, size=[m, n])



G_sample = generator(Xpre, Z)
#D_real, D_logit_real = discriminator(X)
#D_fake, D_logit_fake = discriminator(G_sample)
D_real = discriminator(Xpre, X)
D_fake = discriminator(Xpre, G_sample)
#D_real = discriminator(X)
#D_fake = discriminator(G_sample)
Drift_eq = drift(X)

# D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
# G_loss = -tf.reduce_mean(D_fake)

# Alternative losses:
# -------------------
if beta == 0:
    D_loss_real = tf.reduce_mean(D_real)
else:
    max_val = tf.reduce_max((-beta) * D_real)
    D_loss_real = -(1.0 / beta) * (tf.log(tf.reduce_mean(tf.exp((-beta) * D_real - max_val))) + max_val)

if gamma == 0:
    D_loss_fake = tf.reduce_mean(D_fake)
    G_loss = -tf.reduce_mean(D_fake)

else:
    max_val = tf.reduce_max((gamma) * D_fake)
    D_loss_fake = (1.0 / gamma) * (tf.log(tf.reduce_mean(tf.exp(gamma * D_fake - max_val))) + max_val)
    max_val = tf.reduce_max((gamma) * D_fake)
    G_loss = - (1.0 / gamma) * (tf.log(tf.reduce_mean(tf.exp(gamma * D_fake - max_val))) + max_val)

D_loss = D_loss_real - D_loss_fake

D_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(-D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(G_loss, var_list=theta_G)

#clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in theta_D]
if (beta == 0) & (gamma == 0):
    clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in theta_D]
else:
    clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in theta_D]



toy_data = genfromtxt('V2_data/double_well/double_well_well.csv', delimiter=',', dtype='float32')

#NoT = 500
#idx = np.random.randint(toy_data.shape[0], size=toy_data.shape[0])
#i = int(toy_data.shape[0]) - NoT
#
#toy_data_train = toy_data[idx[:i], :]
#toy_data_test = toy_data[idx[i:], :]

config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if not os.path.exists('V2_results/double_well/csv/'):
    os.makedirs('V2_results/double_well/csv/')

k = 1
SF = 100
D_loss_plots = np.ones(shape=(epochs, 1))
G_loss_plots = np.ones(shape=(epochs, 1))

for it in range(epochs):
    for j in range(k):
        idx = np.random.randint(toy_data.shape[0]-1, size=mb_size)
        Xpre_mb = toy_data[idx,:]
        X_mb = toy_data[idx+1,:]

        _, D_loss_curr, _ = sess.run([D_solver, D_loss, clip_D], feed_dict={
                                         Xpre: Xpre_mb, X: X_mb, Z: sample_Z(mb_size, Z_dim)})

    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={
                                  Xpre: Xpre_mb, X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    if it % SF == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print(sess.run(G_W1))
        print(sess.run(G_W3))
        print(sess.run(sigma))

    D_loss_plots[it] = D_loss_curr
    G_loss_plots[it] = G_loss_curr


NoSamples = 100000
X_ = np.zeros(shape=(NoSamples, X_dim))

Xpre_ = np.random.normal(0., 1, size=[1, X_dim])
for i in range(NoSamples):
    X_tmp = sess.run(G_sample, feed_dict={
                Xpre: Xpre_, Z: sample_Z(1, Z_dim)})
    
    X_[i,:] = X_tmp
    Xpre_ = X_tmp

with open('V2_results/double_well/csv/pathgan_beta_'+str(beta)+'_'+str(iteration)+'_nonpar.csv', "w") as output: # _nonpar
    writer = csv.writer(output, lineterminator='\n')
    for val in X_:
        writer.writerow(val)
        

NoPoints = 1000
V = np.zeros(shape=(NoPoints,NoPoints, X_dim))
X_tmp = np.zeros(shape=(1, X_dim))
for i in range(0, NoPoints):#calculates the V
    for j in range(0, NoPoints):
        X_tmp[0,0] = -5.0+i*dt
        X_tmp[0,1] = -5.0+j*dt
        #print(i,j)
        V_tmp = sess.run(Drift_eq, feed_dict={X: X_tmp})
        
        V[i,j,:] = V_tmp

with open('V2_results/double_well/csv/pathgan_beta_'+str(beta)+'_'+str(iteration)+'_drift_nonpar.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in V:
        writer.writerow(val)


