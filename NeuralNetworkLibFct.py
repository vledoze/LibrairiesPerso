#!/usr/bin/python

# Auteur : vledoze
# Date   : 13/04/2017

import math
import numpy as np

import MachineLearningLibFct as ml

#-----------------------------------------------------------------------
DEBUG = True
if DEBUG:
    import pdb

#-----------------------------------------------------------------------
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
        self.__lst_mat_pond = [np.matrix(np.random.rand(self.__vec_sL[i+1], (self.__vec_sL[i]+1))) for i in range(self.__num_L)]

    ### Methodes ------------------------------------------------------------###
    def mtd_mat_y0(self, vec_y):
        """ Transforme un vecteur de sortie en matrice de sortie pour comparer
            avec les sorties du reseau de neurones
            entree : vec_y   : sorties du training set
            sortie : mat_y0 : sorties du trainig set sous forme de matrice de {0, 1}
        """
        mat_y0 = []
        num_m = len(vec_y)
        for i in range(num_m):
            vec_y0 = []
            for k in range(self.__num_k):
                if vec_y[i] == k+1:
                    vec_y0.append(1.)
                else:
                    vec_y0.append(0.)
            mat_y0.append(vec_y0)
        return mat_y0
        
    def mtd_fwd_activation(self, vec_a, num_iL):
        """ Fonction de calcul des activations dans le sens forward
            entree : vec_a  : vecteur des activations courantes
                     num_iL : numero du layer courant
            sorties: vec_a1 : vecteur des activations suivantes
        """
        mat_theta = self.__lst_mat_pond[num_iL]
        num_na = self.__vec_sL[num_iL+1]
        return [ml.fct_hyp_cls_lin(mat_theta[i,:].A1, vec_a) for i in range(num_na)]

    def mtd_hyp(self, vec_x0):
        """ Fonction hypothese propre au reseau
            entree : vec_x0 : vecteur en entree du reseau de neurones + biais
            sorties: vec_h  : vecteur hypothese en sortie
        """
        vec_a = vec_x0
        for l in range(self.__num_L):
            if l > 0:
                vec_a = np.insert(vec_a, 0, 1.)
            vec_a = self.mtd_fwd_activation(vec_a, l)
        return vec_a

    def mtd_calc_cout(self, mat_x, vec_y, num_lambda=-1):
        """ Calcul de la fonction de cout associee au reseau de neurones
            pour un training set donne
            entree : [mat_x, vec_y] : training set
                     num_lambda     : coefficient overfitting
            sortie : num_J : cout
        """
        num_m = len(vec_y)
        num_J1 = 0.
        num_J2 = 0.
        mat_y0 = self.mtd_mat_y0(vec_y)
        mat_x0 = ml.fct_add_ones(mat_x)
        for m in range(num_m):
            vec_x0 = mat_x0[m].A1
            vec_y0 = mat_y0[m]
            for k in range(self.__num_k):
                num_y0 = vec_y0[k]
                num_h  = self.mtd_hyp(vec_x0)[k]
                num_J1 -= ml.fct_cout_cls(num_y0, num_h)/num_m
        if num_lambda > 0:
            for l in range(self.__num_L):
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
            sortie : lst_mat_dlt    : liste des matrices de gradient du cout
        """
        #Initialisation
        num_m = len(vec_y)
        mat_y0 = self.mtd_mat_y0(vec_y)
        mat_x0  = ml.fct_add_ones(mat_x)
        lst_mat_dlt = []
        for l in range(self.__num_L):
            lst_mat_dlt.append(np.matrix(np.zeros([self.__vec_sL[l+1], self.__vec_sL[l]+1])))
        #Corp
        for m in range(num_m):
            #Forward
            vec_a = mat_x0[m].A1
            lst_vec_a = [vec_a]
            for l in range(self.__num_L):
                vec_a = self.mtd_fwd_activation(vec_a, l)
                if l < (self.__num_L - 1):
                    vec_a.insert(0, 1.)
                lst_vec_a.append(vec_a)
            #Backward
            lst_vec_d = [[(a-out) for a,out in zip(lst_vec_a[self.__num_L], mat_y0[m])]]
            for l in reversed(range(self.__num_L)):
                vec_d = []
                for i in range(1, self.__vec_sL[l]+1):
                    num_dz = 0.
                    for j in range(self.__vec_sL[l+1]):
                        num_dz += lst_vec_d[0][j]*self.__lst_mat_pond[l][j, i] 
                    num_dg = lst_vec_a[l][i]*(1.- lst_vec_a[l][i])
                    vec_d.append(num_dz*num_dg)
                lst_vec_d.insert(0, vec_d)
            #gradient
            for l in range(self.__num_L):
                mat_theta = self.__lst_mat_pond[l]
                for i in range(1, self.__vec_sL[l]+1):
                    for j in range(self.__vec_sL[l+1]):
                        if i == 0 & num_lambda > 0:
                            lst_mat_dlt[l][j, i] += (lst_vec_d[l+1][j] + num_lambda*mat_theta[j, i])/num_m
                        else:
                            lst_mat_dlt[l][j, i] += (lst_vec_d[l+1][j]*lst_vec_a[l][i])/num_m
        return lst_mat_dlt

    def mtd_grad_est(self, mat_x, vec_y, num_eps, num_lambda=-1):
        """ Methode d'estimation du gradient
        """
        lst_mat_grad = []
        num_J_ref = self.mtd_calc_cout(mat_x, vec_y, num_lambda)
        for l in range(self.__num_L):
            mat_theta = self.__lst_mat_pond[l]
            mat_grad = np.matrix(np.zeros([self.__vec_sL[l+1], self.__vec_sL[l]+1]))
            for j in range(self.__vec_sL[l]+1):
                for i in range(self.__vec_sL[l+1]):
                    num_theta_ref = self.__lst_mat_pond[l][i, j]
                    self.__lst_mat_pond[l][i, j] += num_eps
                    num_J_plus = self.mtd_calc_cout(mat_x, vec_y, num_lambda)
                    mat_grad[i, j] = (num_J_plus - num_J_ref)/num_eps
                    self.__lst_mat_pond[l][i, j] = num_theta_ref
            lst_mat_grad.append(mat_grad)
        return lst_mat_grad


    def mtd_train(self, mat_x, vec_y, num_alpha, num_eps, num_lambda=-1):
        """ Methode d'entrainement du reseau de neurones
        """
        boo_repeat = True
        lst_mat_pond_ref =self.__lst_mat_pond
        while boo_repeat:
            lst_vec_dlt = self.mtd_bwd_prpg(mat_x, vec_y, num_lambda)
            lst_mat_pond_tmp = self.__lst_mat_pond
            boo_repeat = False
            for l in range(self.__num_L):
                for j in range(self.__vec_sL[l]+1):
                    for i in range(self.__vec_sL[l+1]):
                        if abs(num_alpha[l]*lst_vec_dlt[l][i, j]) > num_eps:
                            lst_mat_pond_tmp[l][i, j] = self.__lst_mat_pond[l][i, j]- num_alpha[l]*lst_vec_dlt[l][i, j] 
                            boo_repeat = True
            self.__lst_mat_pond = lst_mat_pond_tmp
            if DEBUG :
                num_J = self.mtd_calc_cout(mat_x, vec_y, num_lambda)
                print abs(num_J)
    
    ### Save/Load -----------------------------------------------------------###
    def save(self, str_f):
        file_f = open(str_f, "w")
        for l in range(self.__num_L):
            for i in range(self.__vec_sL[l+1]):
                for j in range(self.__vec_sL[l]+1):
                    file_f.write(str(self.__lst_mat_pond[l][i, j]))
                    file_f.write(",")
                file_f.write(";\n")
            file_f.write("|\n")
        file_f.close()
    
    def load(self, str_f):
        file_f = open(str_f, "r")
        str_data = file_f.read().split("|\n")
        self.__num_L = len(str_data) - 1
        self.__vec_sL = []
        self.__lst_mat_pond = []
        for l in range(self.__num_L):
            str_data_l = str_data[l].split(";\n")
            num_i = len(str_data_l) - 1
            vec_i = []
            for i in range(num_i):
                str_data_i = str_data_l[i].split(',')
                num_j = len(str_data_i) - 1
                vec_j = []
                for j in range(num_j):
                    vec_j.append(float(str_data_i[j]))
                vec_i.append(vec_j)
            self.__lst_mat_pond.append(np.matrix(vec_i))
            self.__vec_sL.append(num_j)
        self.__vec_sL.append(num_i)
        self.__num_n = self.__vec_sL[0]
        self.__num_k = self.__vec_sL[-1]
        file_f.close()
        
    ### Surcharge -----------------------------------------------------------###
    def __str__(self):
        """ Surcharge methode __str__
        """
        return "Matrices de ponderation : {}".format(
                self.__lst_mat_pond)
    
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
    nn_nwk = NeuralNetwork(3, 1, [3, 3, 3])
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
    mat_x = np.matrix([[1., 2., 3.], [4., 5., 6.], [3., 14., 9.], [7., 8., 9.]])
    mat_x0 = ml.fct_add_ones(mat_x)
    vec_y = np.array([1., 0., 1., 0.])
    print "Training set"
    print "Entrees : "
    print mat_x
    print "Sorties : "
    print vec_y
    print "Reseau de neurones"
    for i in range(len(mat_x)):
        print "Estimation echantillon ", i
        print nn_nwk.mtd_hyp(mat_x0[i].A1)
    print "Cout total : ", nn_nwk.mtd_calc_cout(mat_x, vec_y, 1)
    print "Gradient : "
    grad = nn_nwk.mtd_bwd_prpg(mat_x, vec_y)
    for i in range(nn_nwk.get_num_L()):
        print grad[i]
    print "Gradient Estimee : "
    grad_est = nn_nwk.mtd_grad_est(mat_x, vec_y, 0.00001)
    for i in range(nn_nwk.get_num_L()):
        print grad_est[i]
    print "Entrainement"
    nn_nwk.mtd_train(mat_x, vec_y, [1., 1., 0.6, 0.3], 0.0000000000001)
    print "Matrice de ponderation/layer : "
    for i in range(nn_nwk.get_num_L()):
        print "Matrice ", i
        print nn_nwk.get_lst_mat_pond()[i]
    for i in range(len(mat_x)):
        print "Estimation echantillon ", i
        print nn_nwk.mtd_hyp(mat_x0[i].A1)
    print nn_nwk
    nn_nwk.save("save.txt")
    nn_nwk.load("save.txt")
    print nn_nwk
