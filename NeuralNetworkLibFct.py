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
        - __num_n : nombre de classe en entree
        - __num_k : nombre de classe en sortie
        - __num_L : nombre de "layers" du reseau
        - __vec_sL : nombre d'unite par "layer" sans le biais
            vec_sl[0] = nombre d'activation du layer 1
            ...
            vec_sl[__num_L-1] = nombre d'activation du layer __num_L
        - __lst_mat_pond : liste des matrices des ponderations
            lst_mat_pond[0] = matrice permettant de passer des entrees au layer 1
            ...
            lst_mat_pond[i] = matrice permettant de passer du layer i au layer i+1
            ...
            lst_mat_pond[num_L] = matrice permettant de passer du layer num_L aux sorties
    """

    def __init__(self, num_n=1, num_k=1, vec_na=[1]):
        """ Constructeur de la classe NeuralNetwork
            entrees : num_n : nombre de classes en entree
                      num_k : nombre de classes en sortie
                      vec_na : vecteur de repartition des couches d'activation
        """
        self.__num_n = num_n
        self.__num_k = num_k
        self.__vec_sL = np.concatenate(([num_n], vec_na, [num_k]))
        self.__num_L = len(self.__vec_sL) - 1
        self.__lst_mat_pond = [np.matrix(np.ones([self.__vec_sL[i+1], (self.__vec_sL[i]+1)])) for i in range(self.__num_L)]

    ### Methodes ------------------------------------------------------------###
    def mtd_fwd_activation(self, vec_a, num_iL):
        """ Fonction de calcul des activations dans le sens forward
            entree : vec_a  : vecteur des activations courantes
                     num_iL : numero du layer courant
            sorties: vec_a1 : vecteur des activations suivantes
        """
        mat_theta = self.__lst_mat_pond[num_iL]
        num_na = self.__vec_sL[num_iL+1]
        return [ml.fct_hyp_cls_lin(mat_theta[i,:].A1, vec_a) for i in range(num_na)]

    def mtd_bwd_activation(self, vec_a, num_iL):
        """ Fonction de calcul des activations dans le sens backward
            entree : vec_a  : vecteur des activations courantes
                     num_iL : numero du layer courant
            sorties: vec_a1 : vecteur des activations suivantes
        """
        mat_theta = self.__lst_mat_pond[num_iL-1]
        num_na = self.__vec_sL[num_iL-1]
        return [ml.fct_hyp_cls_lin(mat_theta[:,i+1].A1, vec_a) for i in range(num_na)]

    def mtd_hyp(self, vec_x):
        """ Fonction hypothese propre au reseau
        """
        vec_a = vec_x
        for i in range(self.__num_L-1):
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
            for l in range(self.__num_L-1):
                mat_theta = self.__lst_mat_pond[l]
                num_s0 = self.__vec_sL[l]
                num_s1 = self.__vec_sL[l+1]
                for i in range(num_s0):
                    for j in range(num_s1):
                        num_J2 += pow(mat_theta[j,i], 2)
            num_J2 = num_J2*num_lambda/(2.0*num_m)
        return num_J1+num_J2

    def mtd_bwd_prpg(self, mat_x, vec_y, num_lambda=-1):
        """ Algorithme de back-propagation pour calculer la derivee partielle
            du cout du reseau de neurones
            entree : [mat_x, vec_y] : training set
            sortie : num_gradJ      : gradient du cout
        """
        num_m = len(vec_y)
        mat_out = self.mtd_mat_out(vec_y)
        mat_x0  = ml.fct_add_ones(mat_x)
        mat_dlt = np.zeros([self.__num_L-1, num_m, self.__num_n])
        for i in range(num_m):
            #Forward
            vec_a = mat_x0[i].A1
            lst_vec_a = [vec_a]
            for l in range(self.__num_L):
                vec_a = self.mtd_fwd_activation(vec_a, l)
                if l < (self.__num_L - 1):
                    vec_a.insert(0, 1.)
                lst_vec_a.append(vec_a)
            #Backward
            lst_vec_d = [[(a-out) for a,out in zip(lst_vec_a[self.__num_L], mat_out[i])]]
            for l in reversed(range(self.__num_L)):
                vec_dz = self.mtd_bwd_activation(lst_vec_d[0], l+1)
                pdb.set_trace()
                vec_dg = [a*(1. - a) for a in lst_vec_a[l]]
                vec_d = [(dz*dg) for dz,dg  in zip(vec_dz, vec_dg)]
                lst_vec_d.insert(0, vec_d)
            pdb.set_trace()
            for l in range(self.__num_L-1):
                mat_theta = self.__lst_mat_pond[l]
                for jj in range(self.__vec_sL[l]):
                    for ii in range(self.__vec_sL[l+1]):
                        if jj == 0 & num_lambda > 0:
                            mat_dlt[l, i, j] = mat_dlt[l, i, j] + (lst_vec_a[l][jj]*lst_vec_d[l+1][ii] + num_lambda*mat_theta[ii, jj])/num_m
                        else:
                            mat_dlt[l, i, j] = mat_dlt[l, i, j] + (lst_vec_a[l][jj]*lst_vec_d[l+1][ii])/num_m

    def mtd_mat_out(self, vec_y):
        """ Transforme un vecteur de sortie en matrice de sortie pour comparer
            avec les sorties du reseau de neurones
            entree : vec_y   : sorties du training set
            sortie : mat_out : sorties du trainig set sous forme de matrice de {0, 1}
        """
        mat_out = []
        num_m = len(vec_y)
        for i in range(num_m):
            vec_out = []
            for k in range(self.__num_k):
                if vec_y[i] == k+1:
                    vec_out.append(1)
                else:
                    vec_out.append(0)
            mat_out.append(vec_out)
        return mat_out

    ### Setter --------------------------------------------------------------###
    def set_num_L(self, num_L):
        self.__num_L = num_L

    def set_vec_sL(self, vec_sL):
        self.__vec_sL = vec_sL

    def set_num_k(self, num_k):
        self.__num_k = num_k

    def set_lst_mat_pond(self, lst_mat_pond):
        self.__lst_mat_pond = lst_mat_pond

    ### Getter --------------------------------------------------------------###
    def get_num_L(self):
        return self.__num_L

    def get_vec_sL(self):
        return self.__vec_sL

    def get_num_k(self):
        return self.__num_k

    def get_lst_mat_pond(self):
        return self.__lst_mat_pond

if __name__ == "__main__":
    """ Test des fonctions """
    print "Test des fonctionnalites"
    nn_nwk = NeuralNetwork(3)
    print "Creation du reseau de neurones"
    print "Nombre de layers : ", nn_nwk.get_num_L()
    print "Distributions des activations/layers : "
    for i in range(nn_nwk.get_num_L()):
        print "Layer ", i, " - ", nn_nwk.get_vec_sL()[i], "activations"
    print "Nombre de sorties : ", nn_nwk.get_num_k()
    print "Matrice de ponderation/layer : "
    for i in range(nn_nwk.get_num_L()):
        print "Matrice ", i
        print nn_nwk.get_lst_mat_pond()[i]
    mat_x = np.matrix([[1., 2., 3.], [4., 5., 6.], [3., 14., 9.]])
    vec_y = np.array([1., 0., 1.])
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
    nn_nwk.mtd_bwd_prpg(mat_x, vec_y)
