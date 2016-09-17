# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:11:22 2016

@author: spark
"""

import numpy as np
import pylab as pl

# afficher l'image "x" d'indice "index" (au plus &- images) et les sauvegarder si "string" != None:
    #x: la matrice représentant l'image
    #y: le vrai label correspondant à l'image
    # type_donnees: les entrées peuvent etre de type "T.tensor" ou des "numpy"
    # pred: le vecteur de la prediction de x
    # string : le nom du fichier

       
def displayimage(index, x, y, type_donnees = 'tensor', pred = None, string= None):
    
    if (type_donnees == 'tensor'):        
        y = y.eval()
        x = x.get_value()
    
    if (len(index)> 16):
        index = index[0:16]

    for count, i in enumerate(index):
        image = x[i]
        rrows = np.int(np.sqrt(len(index)))
        pl.subplot(rrows, np.int(len(index)/rrows), count+1).imshow(image.reshape(28, 28), cmap=pl.cm.gray)
        pl.title('vrai: %s et predit: %s' % (y[i], pred[i]))   
        pl.axis('off')
        
    if (string != None):
        pl.savefig(string)
    else:
        pl.show()
