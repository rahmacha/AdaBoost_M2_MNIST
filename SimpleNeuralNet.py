# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:36:44 2016

@author: rahma.chaabouni
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'


import numpy as np
import theano
import theano.tensor as T
import time

from logistic_sgd import LogisticRegression


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Couche caché d'un réseau de neurones: les unités sont complétement connectées et ont
        une fonction d'activation tanh. La matrice des poids W est de dimension (n_in,n_out)
        et le vecteur du biais b de dimension (n_out).

        L'activation de l'unité caché est donnée par: tanh(dot(input,W) + b)
        
        Paramètres:
        rng: numpy.random.RandomState -> une nombre aléatoire utilisé pour initialiser les poids
        input: theano.tensor.dmatrix -> tensor symbolique de dimension (n_examples, n_in)
        n_in: int -> dimension de l'entrée
        n_out: int -> nombre d'unités cachées
        activation: theano.Op ou fonction -> fonction d'activation de la couche cachée
        
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Classe de réseau de neurones multicouches
    """

    def __init__(self, rng, input, n_in, n_hiddens, n_out, activations):
        """Initialisation des paramètres du réseau

        Paramètres:
        rng: numpy.random.RandomState -> une nombre aléatoire utilisé pour initialiser les poids
        input: theano.tensor.dmatrix -> tensor symbolique de dimension (n_examples, n_in)
        n_in: int -> dimension de l'entrée
        n_hidden: vecteur d'entiers -> nombre d'unités cachées par couche
        n_out: int -> nombre d'unités de la couche de sortie
        activation: vecteur de theano.Op ou fonction -> fonction d'activation des couches cachées
        """

        self.hiddenLayers = []
        
        self.hiddenLayers.append(HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hiddens[0],
            activation=activations[0]
            ))
            
        for i in range(len(n_hiddens)-1):
            self.hiddenLayers.append(HiddenLayer(
            rng=rng,
            input=self.hiddenLayers[i].output, 
            n_in=n_hiddens[i],
            n_out=n_hiddens[i+1],
            activation=activations[i+1]
            ))
            
        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hiddens[-1],
            n_out=n_out
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        
        self.quadraticLoss = (
            self.logRegressionLayer.quadraticLoss
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        self.incorrects = self.logRegressionLayer.incorrects

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = []
        for hl in self.hiddenLayers:
            self.params = self.params + hl.params
            
        self.params = self.params + self.logRegressionLayer.params
        
        # keep track of model input
        self.input = input
        
        self.y_pred = self.logRegressionLayer.y_pred
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        
        # norme L1, à utiliser pour la régularisation L1
        self.L1 = abs(self.logRegressionLayer.W).sum()
        for hl in self.hiddenLayers:
            self.L1 += abs(hl.W).sum()   
        
        # norme L2 au carré, à utiliser pour la régularisation L2
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()
        for hl in self.hiddenLayers:
            self.L2_sqr += (hl.W ** 2).sum()


    def train(self, weights, train_set_x, train_set_y, x, y, w, reg_L1, reg_L2, 
              learning_rate=0.01, n_epochs=1000, batch_size=20, 
              valid_set_x = None, valid_set_y = None):
        """
        Entrainer un réseau de neurones avec le principe forward/backwark propagation
        et descente de gradient. Si les données de validation sont utilisés,
        l'apprentissage utilisera le principe du "early stopping"
    
        Paramètres
        weights: vecteur de float -> poids associés à chaque exemple d'apprentissage
        train_set_x: theano.tensor.dmatrix -> tensor symbolique des features
        train_set_y: theano.tensor.lvector -> tensor symbolique représentant les labels
        learning_rate: float
        x = T.matrix('x')
        y= T.ivector('y')
        w = T.dvector('w')
        reg_L1:      int -> paramètre de régularisation L1
        reg_L2:      int -> paramètre de régularisation L2
        learning_rate: float -> pas de la descente de gradient
        n_epochs:    int -> nombre d'epochs pour l'entrainement
        batch_size:  int -> taille du batch (si égal à 1 descente stochastique)
        valid_set_x et valid_set_y si pas None appliquer le early stopping
       """
       
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
 
    
        # the cost we minimize during training is the quadratic loss of
        # the model
        cost = (
            self.negative_log_likelihood(y, w) 
            + reg_L1 * self.L1
            + reg_L2 * self.L2_sqr
        )
    
        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
    
        # start-snippet-5
        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in self.params]
        #print([pp(T.grad(cost, param)) for param in self.params])
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
    
        # given two lists of the same length, A = [a1, a2, a3, a4] and
        # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
        # element is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        index = T.lscalar()
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                w: weights[index * batch_size: (index + 1) * batch_size]
            }
        )
        
        if(valid_set_x != None):
            n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
            
            validate_model = theano.function(
                inputs=[index],
                outputs=self.errors(y, w),
                givens={
                    x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                    y: valid_set_y[index * batch_size:(index + 1) * batch_size],
                    w: np.ones(batch_size)
                }
            )
        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        
        if ((valid_set_x == None) or (valid_set_y == None)):
            for epoch in range(n_epochs):
                #tstart = time.time()
                for minibatch_index in range(n_train_batches):
                    #print(weights.get_value(borrow = True)[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])
                    #print(sum(weights.get_value(borrow = True)[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]))
                    
                    minibatch_avg_cost = train_model(minibatch_index)
                #print('temps d epoch', time.time()-tstart)
        else:
            patience = 10000  # look as this many examples regardless
            patience_increase = 2  # wait this much longer when a new best is
                                   # found
            improvement_threshold = 0.995  # a relative improvement of this much is
                                           # considered significant
            validation_frequency = min(n_train_batches, patience // 2)
                                          # go through this many
                                          # minibatche before checking the network
                                          # on the validation set; in this case we
                                          # check every epoch
        
            best_validation_loss = np.inf
            best_iter = 0
            
            epoch = 0
            done_looping = False
        
            while (epoch < n_epochs) and (not done_looping):
                epoch = epoch + 1
                for minibatch_index in range(n_train_batches):
                    minibatch_avg_cost = train_model(minibatch_index)
                    # iteration number
                    iter = (epoch - 1) * n_train_batches + minibatch_index
        
                    if (iter + 1) % validation_frequency == 0:
                        # compute zero-one loss on validation set
                        validation_losses = [validate_model(i) for i
                                             in range(n_valid_batches)]
                        this_validation_loss = np.mean(validation_losses)
        
    #                    print(
    #                        'epoch %i, minibatch %i/%i, validation error %f %%' %
    #                        (
    #                            epoch,
    #                            minibatch_index + 1,
    #                            n_train_batches,
    #                            this_validation_loss * 100.
    #                        )
    #                    )
        
                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:
                            #improve patience if loss improvement is good enough
                            if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                            ):
                                patience = max(patience, iter * patience_increase)
        
                            best_validation_loss = this_validation_loss
                            best_iter = iter
        
                    if patience <= iter:
                        done_looping = True
                        break
            print(("Fin de l'optimisation. Meilleur score de validation  %f %% "
                   'obtained at iteration %i') %
                  (best_validation_loss * 100., best_iter + 1))   
    
    def predict(self, new_x):
        """
        Appliquer le réseau appris sur "new_x" pour prédir ses labels        
        Paramètres
        new_x: matrice numpy -> features 
       """
        predictions = theano.function(inputs = [self.input],
                                      outputs = self.y_pred)    
        return predictions(new_x)
        
    def score(self, new_x):
        """
        Appliquer le réseau appris sur "new_x" pour le score de chaque labels        
        Paramètres
        new_x: matrice numpy -> features 
       """        
        predictions = theano.function(inputs = [self.input],
                                      outputs = self.p_y_given_x)    
        return predictions(new_x)
