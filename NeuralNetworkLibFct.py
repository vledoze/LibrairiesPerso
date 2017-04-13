#!/usr/bin/python

# Auteur : vledoze
# Date   : 13/04/2017

import math
import numpy as np

import MachineLearningLibFct

class NeuralNetwork:
    """ Classe NeuralNetwork
        Defini un reseau de neurone avec les attributs suivant
        - __num_L : nombre de "layers" du reseau
        - __num_k : nombre de classe de sortie
        - __vec_sL : nombre d'unite par "layer" sans le biais
            vec_sl[0] = nombre d'entrees
            ...
            vec_sl[num_L] = num_k
        - __mat_pond : matrice des ponderations
            mat_pond[0] = matrice permettant de passer du layer 0 au layer 1
    """

    def __init__(self, num_L=2, vec_sL=[1, 1], num_k=1):
        """ Constructeur de la classe NeuralNetwork
        """
        self.__num_L = num_L + 1
        self.__num_k = num_k
        self.__vec_sL = vec_sL
        self.__vec_sL.append(num_k)
        for i in range(num_L-1):
            mat_theta0 = np.ones(vec_sL[i+1], (vec_sL[i]+1))
            #TODO ajouter random.gauss sur la matrice theta
            self.__mat_pond.append(mat_theta0)

    ### Methodes ------------------------------------------------------------###
    def mtd_hyp(self, vec_x):
        """ Fonction hypothese propre au reseau
        """
        num_n = len(vec_x) + 1
        vec_a = vec_x
        for i in range(self.__num_L - 1):
            vec_a1 = [1]
            for j in range(self.__vec_sL[i]):
                vec_theta = self.__mat_theta[i][j]
                vec_a_1.append(fct_hyp_cls_lin(vec_theta, vec_a))
            vec_a = vec_a1
        return vec_a

    def mtd_calc_cout(self, mat_x, vec_y):
        """ Calcul de la fonction de cout associee au reseau de neurones
            pour un training set donne
            entree : [mat_x, vec_y] : training set
            sortie : num_J : cout
        """
        num_m = len(vec_y)
        mat_x0 = fct_add_ones(mat_x)
        num_J1 = 0
        for i in range(num_m):
            vec_x = mat_x0[i].A1
            for k in range(self.num_k):
                if vec_y(i)==k:
                    num_y = 1
                else:
                    num_y = 0
                num_h = self.mtd_hyp(vec_x)
                num_J1 += fct_cout_cls(num_y, num_h)
                #TODO finir
        return num_J

    ### Setter --------------------------------------------------------------###
    def set_num_L(self, num_L):
        self.__num_L = num_L

    def set_vec_sL(self, vec_sL):
        self.__vec_sL = vec_sL

    def set_num_k(self, num_k):
        self.__num_k = num_k

    ### Getter --------------------------------------------------------------###
    def get_num_L(self):
        return self.__num_L

    def get_vec_sL(self):
        return self.__vec_sL

    def get_num_k(self):
        return self.__num_k

if __name__ == "__main__":
    """ Test des fonctions """
    print "Test des fonctionnalites"
