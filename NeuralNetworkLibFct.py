#!/usr/bin/python

# Auteur : vledoze
# Date   : 13/04/2017

import math
import numpy as np

import MachineLearningLibFct as ml

import pdb

class NeuralNetwork:
    """ Classe NeuralNetwork
        Defini un reseau de neurone avec les attributs suivant
        - __num_L : nombre de "layers" du reseau
        - __num_k : nombre de classe de sortie
        - __vec_sL : nombre d'unite par "layer" sans le biais
            vec_sl[0] = nombre d'activation du layer 1
            ...
            vec_sl[__num_L-1] = nombre d'activation du layer __num_L
        - __mat_pond : matrice des ponderations
            mat_pond[0] = matrice permettant de passer des entrees au layer 1
            ...
            mat_pond[i] = matrice permettant de passer du layer i au layer i+1
            ...
            mat_pond[num_L] = matrice permettant de passer du layer num_L aux sorties
    """

    def __init__(self, num_n=1, num_L=1, num_k=1, vec_sL=[1]):
        """ Constructeur de la classe NeuralNetwork
        """
        self.__num_n = num_n
        self.__num_L = num_L
        self.__num_k = num_k
        self.__vec_sL = vec_sL
        #TODO ajouter random.gauss sur la matrice theta
        mat_theta_0 = np.ones([vec_sL[0], (num_n+1)])
        self.__mat_pond = [mat_theta_0]
        if num_L > 1:
            for i in range(1, num_L-1):
                mat_theta_i = np.ones([vec_sL[i+1], (vec_sL[i]+1)])
                self.__mat_pond.append(mat_theta_i)
        mat_theta_L = np.ones([num_k, (vec_sL[num_L-1]+1)])
        self.__mat_pond.append(mat_theta_L)

    ### Methodes ------------------------------------------------------------###
    def mtd_fwd_activation(self, vec_a, num_iL):
        """ Fonction de calcul des activations dans le sens forward 
            entree : vec_a  : vecteur des activations courantes 
                     num_iL : numero du layer courant
            sorties: vec_a1 : vecteur des activations suivantes
        """
        mat_theta = self.__mat_pond[num_iL]
        vec_a1 = []
        for i in range(len(mat_theta[:, 0])):
            vec_theta = mat_theta[i,:]
            vec_a1.append(ml.fct_hyp_cls_lin(vec_theta, vec_a))
        return vec_a1
            
    def mtd_hyp(self, vec_x):
        """ Fonction hypothese propre au reseau
        """
        vec_a = vec_x
        for i in range(self.__num_L):
            vec_a = self.mtd_fwd_activation(vec_a, i)
        return vec_a

    def mtd_calc_cout(self, mat_x, vec_y, num_lambda=-1):
        """ Calcul de la fonction de cout associee au reseau de neurones
            pour un training set donne
            entree : [mat_x, vec_y] : training set
                     num_lambda     : coefficient overfitting
            sortie : num_J : cout
        """
        num_m = len(vec_y)
        mat_x0 = ml.fct_add_ones(mat_x)
        num_J1 = 0.
        num_J2 = 0.
        for i in range(num_m):
            vec_x = mat_x0[i].A1
            for k in range(self.__num_k):
                if vec_y[i]==k:
                    num_y = 1
                else:
                    num_y = 0
                num_h = self.mtd_hyp(vec_x)[k]
                num_J1 += ml.fct_cout_cls(num_y, num_h)
        num_J1 = -num_J1/num_m
        if num_lambda > 0:
            for l in range(self.__num_L+1):
                mat_theta = self.__mat_pond[l]
                l0 = np.shape(mat_theta)
                for i in range(l0[0]):
                    for j in range(l0[1]):
                        num_J2 += pow(mat_theta[i][j], 2)
            num_J2 = num_J2*num_lambda/(2.0*num_m)
        return num_J1 + num_J2
    
    def mtd_bck_prp(self, mat_x, vec_y):
        """ Algorithme de back-propagation pour calculer la derivee partielle
            du cout du reseau de neurones
            entree : [mat_x, vec_y] : training set
            sortie : num_gradJ      : gradient du cout
        """
        num_m = len(vec_y)
        for i in range(num_m):
            vec_a = mat_x[i]
            mat_a = [vec_a_i]
            for j in range(self.__num_L + 1):
                vec_a = mtd_fwd_activation(vec_a, j)
                mat_a.append(vec_a)
            num_dlt = vec_a
            for j in range(self.__num_L + 1):
                
                

    ### Setter --------------------------------------------------------------###
    def set_num_L(self, num_L):
        self.__num_L = num_L

    def set_vec_sL(self, vec_sL):
        self.__vec_sL = vec_sL

    def set_num_k(self, num_k):
        self.__num_k = num_k
    
    def set_mat_pond(self, mat_pond):
        self.__mat_pond = mat_pond

    ### Getter --------------------------------------------------------------###
    def get_num_L(self):
        return self.__num_L

    def get_vec_sL(self):
        return self.__vec_sL

    def get_num_k(self):
        return self.__num_k
    
    def get_mat_pond(self):
        return self.__mat_pond

if __name__ == "__main__":
    """ Test des fonctions """
    print "Test des fonctionnalites"
    nn_nwk = NeuralNetwork(3)
    print "Creation du reseau de neurones"
    print "Nombre de layers : ", nn_nwk.get_num_L()
    print "Distributions des activations/layers : "
    for i in range(nn_nwk.get_num_L()):
        print "Layer ", i+1, " - ", nn_nwk.get_vec_sL(), "activations"
    print "Nombre de sorties : ", nn_nwk.get_num_k()
    print "Matrice de ponderation/layer : "
    for i in range(nn_nwk.get_num_L()+1):
        print "Matrice ", i+1
        print nn_nwk.get_mat_pond()[i]
    mat_x = np.matrix([[1., 2., 3.], [4., 5., 6.], [3., 14., 9.]])
    vec_y = np.array([1., 0., 2.])
    print "Training set"
    print "Entrees : "
    print mat_x
    print "Sorties : "
    print vec_y
    print "Reseau de neurones"
    for i in range(len(mat_x)):
        print "Estimation echantillon ", i
        print nn_nwk.mtd_hyp(mat_x[i].A1)
    print "Cout total : ", nn_nwk.mtd_calc_cout(mat_x, vec_y, 1)
