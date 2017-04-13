#!/usr/bin/python

# Auteur : vledoze
# Date   : 06/04/2017

import math
import numpy as np

import pdb

# Fonctions hypothese --------------------------------------------------

def fct_hyp_reg_lin(vec_theta, vec_x):
    """ Fonction hypothese pour regression lineaire
        entree : theta = coefficients de la fonction
                 x     = variables de la fonction
        sortie : h(theta, x) = theta(0).x(0) + ... + theta(n).x(n)
    """
    h = 0.0
    for i in range(len(vec_x)):
        h = h + vec_theta[i]*vec_x[i]
    return h

def fct_hyp_cls_lin(vec_theta, vec_x):
    """ Fonction hypothese pour classification
        entree : theta = coefficients de la fonction
                 x     = variables de la fonction
        sortie : hypothese de classification
    """
    return fct_hyp_sigmoid(fct_hyp_reg_lin(vec_theta, vec_x))

def fct_hyp_sigmoid(num_z):
    """ Fonction sigmoide
        entree : z = variable
        sortie : fonction sigmoid [0, 1]
    """
    return 1/(1 + exp(-num_z))

# Fonctions pre-traitement ---------------------------------------------

def fct_feat_scl(vec_x):
    """ Fonction de feature scaling
        entree : x    = variable
        sortie : x(i) = (x(i) - mean(x))/(max(x) - min(x))
                    compris entre [-0.5; 0.5]
    """
    num_mean = np.mean(vec_x)
    num_max = max(vec_x)
    num_min = min(vec_x)
    for i in range(len(vec_x)):
        vec_x[i] = (vec_x[i]-num_mean)/(num_max-num_min)
    return vec_x

def fct_mean_norm(vec_x):
    """ Fonction de mean normalization
        entree : x    = variable
        sortie : x(i) = (x(i) - mean(x))/std(x)
                     compris entre [-1; 1]
    """
    num_mean = np.mean(vec_x)
    num_std = np.std(vec_x)
    for i in range(len(vec_x)):
        vec_x[i] = (vec_x[i]-num_mean)/num_std
    return vec_x

def fct_add_ones(mat_x):
    num_n = len(mat_x[:, 0].A1)
    return np.insert(mat_x, 0, np.ones(num_n), axis=1)

# Fonctions cout -------------------------------------------------------

def fct_cout_reg(num_y, num_h):
    """ Fonction de cout pour une regression
        entree : num_y = une sortie de reference
                 num_h = valeur de l'hypothese obtenue
        sortie : cout de la fonction hypothese / realite
    """
    return pow((num_h - num_y), 2)

def fct_sum_cout_reg(fct_f, vec_theta, mat_x, vec_y):
    """ Fonction de somme de cout pour une regression
        entree : fct_f          = fonction hypothese
                 vec_theta      = coeficients fonction hypothese
                 [mat_x, vec_y] = training set
        sortie : cout total de la fonction hypothese / realite
    """
    num_m = len(vec_y)
    mat_x0 = fct_add_ones(mat_x)
    num_J = 0
    for i in range(num_m):
        vec_x = mat_x0[i, :]
        num_y = vec_y[i]
        num_h = fct_f(vec_theta, vec_x)
        num_J = num_J + fct_cout_reg(num_y, num_h)
    num_J = (1/2*m)*num_J
    return num_J

def fct_cout_cls(num_y, num_h):
    """ Fonction de cout pour une classification
        entree : num_y = une sortie de reference
                 num_h = valeur de l'hypothese obtenue
        sortie : cout de la fonction hypothese / realite
    """
    return -num_y*log(num_h) - (1.0-num_y)*log(1.0-num_h)

def fct_sum_cout_cls(fct_f, vec_theta, mat_x, vec_y):
    """ Fonction de somme de cout pour une regression
        entree : fct_f          = fonction hypothese
                 vec_theta      = coeficients fonction hypothese
                 [mat_x, vec_y] = training set
        sortie : cout total de la fonction hypothese / realite
    """
    num_m = len(vec_y)
    mat_x0 = fct_add_ones(mat_x)
    num_J = 0
    for i in range(num_m):
        vec_x = mat_x0[i, :]
        num_h = fct_f(vec_theta, vec_x)
        num_y = vec_y[i]
        num_J = num_J + fct_cout_cls(num_y, num_h)
    num_J = (1/m)*num_J
    return num_J

# Fonctions principales ------------------------------------------------

def fct_grd_des_reg_lin(mat_x, vec_y, num_alpha, num_eps, num_lambda=-1):
    """ Fonction de descente de gradient pour une regression avec
        fonction hypothese lineaire
        entree : [mat_x, vec_y] = training set
                 num_alpha      = pas de la descente
                 num_eps        = critere de convergence
                 num_lambda     = parametre de regularisation
        sortie : coefficients du modele
    """
    return fct_grd_des(mat_x, vec_y, fct_hyp_reg_lin, num_alpha, num_eps, num_lambda)

def fct_grd_des_cls_lin(mat_x, vec_y, num_alpha, num_eps, num_lambda=-1):
    """ Fonction de descente de gradient pour une classification avec
        fonction hypothese lineaire
        entree : [mat_x, vec_y] = training set
                 num_alpha      = pas de la descente
                 num_eps        = critere de convergence
                 num_lambda     = parametre de regularisation
        sortie : coefficients du modele
    """
    return fct_grd_des(mat_x, vec_y, fct_hyp_cls_lin, num_alpha, num_eps, num_lambda)

def fct_grd_des(mat_x, vec_y, fct_f, num_alpha, num_eps, num_lambda=-1):
    """ Fonction de descente de gradient pour une fonction hypothese quelconque
        entree : [mat_x, vec_y] = training set
                 fct_f          = fonction hypothese
                 num_alpha      = parametre pas de la descente
                 num_eps        = parametre critere de convergence
                 num_lambda     = parametre de regularisation
        sortie : vec_theta      = coefficients du modele
    """
    num_n = len(mat_x[0].A1)+1
    num_m = len(vec_y)
    vec_theta = np.ones(num_n)
    mat_x0 = fct_add_ones(mat_x)
    boo_repeat = True
    while boo_repeat:
        vec_theta_tmp = np.ones(num_n)
        boo_repeat = False
        for j in range(num_n):
            num_J = 0
            for i in range(num_m):
                vec_x = mat_x0[i].A1
                num_J += (fct_f(vec_theta, vec_x) - vec_y[i])*vec_x[j]
            if (num_lambda>0) & (j>0):
                vec_theta_tmp[j] = (  vec_theta[j]*(1.0 - num_alpha*num_lambda/num_m)
                                    - (num_alpha/num_m)*num_J)
            else:
                vec_theta_tmp[j] = vec_theta[j] - (num_alpha/num_m)*num_J
            if abs(vec_theta_tmp[j] - vec_theta[j]) > num_eps:
                boo_repeat = True
        vec_theta = vec_theta_tmp
    return vec_theta

def fct_norm_eq(mat_x, vec_y, num_lambda=-1):
    """ Fontion de calcul des coefficients theta a travers l'equation normale
        entree : [mat_x, vec_y] = training set
                 num_lambda     = parametre de regularisation
        sortie : vec_theta      = coefficients du modele
    """
    mat_x0 = fct_add_ones(mat_x)
    if num_lambda > 0:
        mat_x1_1 = np.dot(mat_x0.T, mat_x0)
        num_n = len(mat_x[0].A1)+1
        mat_L = np.identity(num_n)
        mat_L[0, 0] = 0
        mat_x1_2 = np.dot(num_lambda, mat_L)
        mat_x1 = np.add(mat_x1_1, mat_x1_2)
    else:
        mat_x1 = np.dot(mat_x0.T, mat_x0)
    mat_x2 = np.dot(np.linalg.pinv(mat_x1), mat_x0.T)
    return (np.dot(mat_x2, vec_y)).A1

if __name__ == "__main__":
    """ Test des fonctions """
    mat_x = np.matrix([[1., 2., 3.], [4., 5., 6.], [3., 14., 9.]])
    mat_x0 = fct_add_ones(mat_x)
    vec_y = np.array([1., 3., 7.])

    #print fct_feat_scl(vec_y)
    #print fct_mean_norm(vec_y)
    print "Training set"
    print "Entrees : "
    print mat_x
    print "Sorties : "
    print vec_y
    print "Test descente de Gradient"
    num_theta1 = fct_grd_des_reg_lin(mat_x, vec_y, 0.001, 0.00001, 0)
    print "Coefficients  : ",  num_theta1
    print "Estimation obtenue : "
    print fct_hyp_reg_lin(num_theta1, mat_x0[0, :].A1), \
          fct_hyp_reg_lin(num_theta1, mat_x0[1, :].A1), \
          fct_hyp_reg_lin(num_theta1, mat_x0[2, :].A1)
    print "Test equation normale"
    num_theta2 = fct_norm_eq(mat_x, vec_y, 0)
    print "Coefficients : ", num_theta2
    print "Estimation obtenue : "
    print fct_hyp_reg_lin(num_theta2, mat_x0[0, :].A1), \
          fct_hyp_reg_lin(num_theta2, mat_x0[1, :].A1), \
          fct_hyp_reg_lin(num_theta2, mat_x0[2, :].A1)
