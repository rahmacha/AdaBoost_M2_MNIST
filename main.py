# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:35:51 2016

@author: rahma.chaabouni
"""
from __future__ import division

import os
from logistic_sgd import load_data
import theano
import theano.tensor as T
import numpy as np
import time
import pandas as pd

from AdaM2 import AdaBoost_M2, Hypothesis
from Plot_Digits import displayimage


os.chdir('/home/spark/Documents/Recherches/Theano/Datasets')
dataset='mnist.pkl.gz'
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[2]

x_train = train_set_x.get_value(borrow = True)
Y_true_train = train_set_y.eval()

x_test = test_set_x.get_value(borrow = True)
Y_true_test = test_set_y.eval()


T#_max = (1,2,3,5,10,20,100,500,1000,2000)
T_max = (1,10000)

er_train = []
er_test = []
# allocate symbolic variables for the data
t0 = time.time()
for t in T_max:
    H = AdaBoost_M2(train_set_x,train_set_y, t ,reg_L1 = 0.0, reg_L2 = 0.0, learning_rate=0.01,
                 n_epochs=10, batch_size=20, n_in = 28*28, 
                 n_hiddens = [10,10], n_out = 10, activations = [T.nnet.relu, T.nnet.relu],
                 x = T.matrix('x'), y= T.ivector('y'), w = T.dvector('w'),
                 valid_set_x = None, valid_set_y = None)
    t1 = time.time()
    print('temps pour AdaBoostM2', t1 - t0)
        
    #Evaluation
    #apprentissage             
    predictions_train = Hypothesis(H[0], H[1], x_train)
    erreur_train = np.sum(predictions_train != Y_true_train)/len(Y_true_train)
    er_train.append(erreur_train)
    
    #test
    predictions_test = Hypothesis(H[0], H[1], x_test)
    erreur_test = np.sum(predictions_test != Y_true_test)/len(Y_true_test)
    er_test.append(erreur_test)

#######################################################################
##                    Changement par iteration                       ##
#######################################################################
changement = '10_2_relu_5000'

#######################################################################
##               Sauvegarder les resultats des erreurs               ##
#######################################################################
df_erreur = pd.DataFrame({'train': er_train, 'test': er_test})
path_erreur = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+ 'erreur_tanh_2000_' + changement
df_erreur.to_csv(path_erreur, index = False)

#######################################################################
##                 afficher les digits mal classées                  ##
#######################################################################

false_pos_train = np.where(predictions_train != Y_true_train)[0]
false_label_train = [predictions_train[x] for x in false_pos_train]
df_train = pd.DataFrame({'pos': false_pos_train, 'label': false_label_train})
path_1 = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+ 'digit_tanh_2000_' + changement
df_train.to_csv(path_1, index = False)
# df_train = pd.read_csv(path_1)

false_pos_test = np.where(predictions_test != Y_true_test)[0]
false_label_test = [predictions_test[x] for x in false_pos_test]
df_test = pd.DataFrame({'pos': false_pos_test, 'label': false_label_test})
path_2 = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+ 'digit_tanh_2000_' + changement + '_test'
df_test.to_csv(path_2, index = False)
# df_train = pd.read_csv(path_2)

#######################################################################
##                 afficher les matrices de confusion                ##
#######################################################################
from sklearn.metrics import confusion_matrix
print('on est la')
string_1 = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+"heatmap_tanh_2000_"+changement+ '.csv'
string_2 = '/home/spark/Documents/Recherches/Rapport/Resultat_AdaM2/'+ "heatmap_tanh_2000_"+changement+"_test.csv"

CM_train = confusion_matrix(Y_true_train, predictions_train)
np.savetxt(string_1, CM_train)

CM_test = confusion_matrix(Y_true_test, predictions_test)
np.savetxt(string_2, CM_test)