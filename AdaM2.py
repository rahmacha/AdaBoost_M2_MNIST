# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 16:34:30 2016

@author: rahma.chaabouni
"""
from __future__ import division

import time
import numpy as np
import theano
import theano.tensor as T
from SimpleNeuralNet import MLP

# AdaBoost Algorithm
def AdaBoost_M2(train_set_x,train_set_y, T_max,reg_L1 = 0.0, reg_L2 = 0.0, learning_rate=0.01,
             n_epochs=10, batch_size=20, n_in = 28*28, 
             n_hiddens = [10, 10], n_out = 10, activations = [T.tanh, T.tanh],
             x = T.matrix('x'), y= T.ivector('y'), w = T.dvector('w'),
             valid_set_x = None, valid_set_y = None):
    k = n_out
    # Initialisation
    m = train_set_x.get_value(borrow = True).shape[0]
    n = np.int(m/batch_size)
    B = m*(k-1)
    weaks = []
    betas = []
    
    weights_part = w/w.sum() 
    normalize_function = theano.function(
            inputs=[w],
            outputs=weights_part)
            
            
    D = (1/B)*np.ones((m,k-1))
    # Induire une distribution pour tout les exemples
    w_tmp = np.sum(D,axis=1)
    w_tmp = w_tmp/w_tmp.sum() 
    
    w_tmp2 = (w_tmp*m)/batch_size
    
    weights_all = theano.shared(w_tmp)
    weights = theano.shared(w_tmp2)
    
    rng = np.random.RandomState(1234)
    
    
    # Construire les T_max classifieurs de bases
    for t in range(T_max):
        print(t)                
                
        h = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hiddens=n_hiddens,
        n_out=n_out,
        activations = activations
        )

        # Entrainer le classifieur de base sur la nouvelle distribution
        
        h.train(weights, train_set_x, train_set_y, x,y,w, reg_L1, reg_L2, 
                learning_rate=learning_rate, n_epochs=n_epochs, 
                batch_size=batch_size, valid_set_x = valid_set_x, valid_set_y = valid_set_y)
                
        # Compute the error
        predictions_proba = h.score(train_set_x.get_value()) # y_predict est un vecteur où chaque composante k représente le degré de confiance à la prediction y_i = k
        # Error fraction
        t0 = time.time()        
        label_train = train_set_y.eval()     
        tmp = 0
        for i in range(m):
            index = 0
            for cl in range(k):
                if(cl !=label_train[i]):
                    tmp += D[i, index]*(1 - predictions_proba[i,label_train[i]] + predictions_proba[i,cl])
                    index +=1 

        epsilon = 0.5* tmp
        print('la psoeudo erreur est', epsilon)
        #print epsilon
        beta = epsilon/(0.00000001+ 1- epsilon)

        # mise à jour de la distribution D_t
        t0 =time.time()
        for i in range(m):
            index = 0
            for cl in range(k):
                if(cl !=label_train[i]):
                    exposant = 0.5*(1 + predictions_proba[i,label_train[i]] - predictions_proba[i,cl])
                    D[i,index] = D[i, index]*beta**exposant
                    index = index+1

        #normaliser la matrice pour avoir une distibution
        D = D/sum(sum(D))
        
        #print 'e=%.2f a=%.2f'%(epsilon, alpha)
        # Compute weights in step t
        t0 = time.time()
        w_tmp = np.sum(D,axis=1)
        w_tmp_nor = w_tmp / w_tmp.sum()
        weights_all.set_value(w_tmp_nor)
        
#        print('temps pour calculer les poids all est', time.time()-t0)
        # Reinitialiser les weights for the minibatch gradient descent
        t0 = time.time()
        new_w_batch = []
        for indice in range(n):
            new_tmp = normalize_function(weights_all.get_value()[indice*batch_size: (indice+1)*batch_size])
            new_w_batch.append(new_tmp)            

        new_w_batch = [item for sublist in new_w_batch for item in sublist]
        weights.set_value(new_w_batch)
   
        weaks.append(h)
        betas.append(beta)
    return (betas, weaks)

# Built the stong classifier
def Hypothesis(betas, learners, x):
    H_x = [np.log(1/beta) * h.score(x) for beta, h in zip(betas, learners)]
    H_x = np.sum(H_x, axis = 0)
    return [np.argmax(xx) for xx in H_x]
