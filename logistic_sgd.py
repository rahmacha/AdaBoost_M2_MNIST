# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:39:29 2016

@author: rahma.chaabouni
"""
from __future__ import division


"""
  La régression logistique est utilisée pour la couche de sortie du réseau
  de neurones. 
  L'équation de ce modèle est:
  y_{pred} = argmax_i P(Y=i|x,W,b)

"""
import numpy as np
import theano
import theano.tensor as T
import os

class LogisticRegression(object):
    """
    Les paramètres de la régression:
    'W': poids
    'b' biais
    """

    def __init__(self, input, n_in, n_out):
        """ Initialisation des paramètres
        
        input: theano.tensor.TensorType -> variable simbolique représentant 
               les entrées
        n_in: int -> nombre des unités de la couche d'entrée
        n_out: int -> nombre des unités de la couche de sortie
        
        """
        
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b) 
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y,w):
        """Retourner la moyenne du "negative log-likelihood" des prédictions

        y: theano.tensor.TensorType -> Les vrais labels
        w: theano.tensor.TensorType -> Les poids associés à chaque label (
           définir la distribution)
        """
 
        return -T.sum(w*T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def quadraticLoss(self, y, w):
        
        return T.sum(w*((self.y_pred -y)*(self.y_pred -y)))

    def errors(self, y,w):
        """Retourner l'erreur de classification des prédictions
        y: theano.tensor.TensorType -> Les vrais labels
        w: theano.tensor.TensorType -> Les poids associés à chaque label (
           définir la distribution)
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(w*T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
            
    def incorrects(self, y):
        """Retourner a vector 0/1 for false or true predictions

            y: theano.tensor.TensorType -> Les vrais labels        
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.neq(self.y_pred, y)
        else:
            raise NotImplementedError()

        
def displayimage(index, x, y, type_donnees = 'tensor', pred = None, string= None):
    """ Afficher les images x corresondant aux indices "index"
        index: vecteur d'entiers -> les indices des exemples à afficiher
        x: vecteur de matrices -> la matrice des features
        y: vecteur d'entier -> Les vrais labels
        type_donnees: string -> si 'tensor' x et y sont des theano.tensor, sinon
                      des numpy
        pred: vecteur -> Si pred != None, la prédiction et la vrai valeur seront 
                         affichés
        string: string -> Si string != None l'image sera enregistrée en la nommant
    """
    
    
    import pylab as pl
    
    if (type_donnees == 'tensor'):        
        y = y.eval()
        x = x.get_value()
    
    if (len(index)> 16):
        index = index[0:16]

    for count, i in enumerate(index):
        image = x[i]
        rrows = np.int(np.sqrt(len(index)))
        pl.subplot(rrows, np.int(len(index)/rrows), count+1).imshow(image.reshape(28, 28), cmap=pl.cm.gray)
        pl.title('vrai: %s et prédit: %s' % (y[i], pred[i]))   
        pl.axis('off')
        
    if (string != None):
        pl.savefig(string)
    else:
        pl.show()
    
def compute_error_confMatrix(df, normalisation):
    sum_diag = sum(np.diag(df))
    return (1 - (sum_diag/normalisation))
    
def save_object(obj, filename):
    import pickle
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    
def display_marge(marge, color, ax):
    values, base = np.histogram(marge, bins=40)
    cumulative = np.cumsum(values)
    ax.plot(base[:-1], cumulative, color = color)
#    return ax
  

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present

    import cPickle, gzip
    os.chdir('/home/spark/Documents/Recherches/Theano/Datasets')
    
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    print('... loading data')
    
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.
        
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(value = np.asarray(data_x, dtype = theano.config.floatX), borrow = borrow)
        shared_y = theano.shared(value = np.asarray(data_y, dtype = theano.config.floatX), borrow = borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y),(valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval 