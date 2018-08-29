#!/usr/bin/python
# coding : utf-8

import math

def prompt_calcul_mensualite():
    """
    """
    print "Calcul Mensualite ==========================================="
    print "Somme empruntee : ",
    emprunt = float(raw_input())
    print "Nombre d'annees : ",
    nb_mois = int(raw_input())*12
    print "Taux d'interet  : ",
    taeg = float(raw_input())/100.0
    print "Taux assurance  : ",
    taux_ass = float(raw_input())
    #Mensualites
    calcul_mensualite(emprunt, nb_mois, taeg, taux_ass)
    
def prompt_simulation_achat():
    """
    """
    print "Simulation d'achat =========================================="
    print "Prix appartement : ",
    prix = float(raw_input())
    nota = calcul_frais_notaire(prix, recent=False) #TODO ajout "recent"
    total = prix + nota
    print "Apport            : ",
    apport = float(raw_input())
    emprunt = total - apport
    simu = True
    while simu:
        print "Emprunt -----------------------------------------------------"
        print "Somme a emprunter : {0}".format(emprunt)
        print "Nombre d'annees   : ",
        nb_mois = int(raw_input())*12
        print "Taux d'interet    : ",
        taeg = float(raw_input())/100.0
        print "Taux assurance    : ",
        taux_ass = float(raw_input())
        #Mensualites
        calcul_mensualite(emprunt, nb_mois, taeg, taux_ass, ammorti=False)
        #Boucle
        print "Cela vous convient-il ? [Oui / Non]:",
        resp = str(raw_input())
        if ((resp == "Oui") | (resp == "") | (resp == "O")):
            print "Fin."
            simu = False
    
def prompt_simulation_achat_revente():
    """
    """
    print "Simulation d'achat avec revente ============================="
    print "Prix appartement : ",
    prix = float(raw_input())
    nota = calcul_frais_notaire(prix, recent=False) #TODO ajout "recent"
    total = prix + nota
    print "Apport            : ",
    apport = float(raw_input())
    emprunt = total - apport
    simu = True
    while simu:
        print "Emprunt -----------------------------------------------------"
        print "Somme a emprunter : {0}".format(emprunt)
        print "Nombre d'annees   : ",
        nb_mois = int(raw_input())*12
        print "Taux d'interet    : ",
        taeg = float(raw_input())/100.0
        print "Taux assurance    : ",
        taux_ass = float(raw_input())
        #Mensualites
        calcul_mensualite(emprunt, nb_mois, taeg, taux_ass, ammorti=False)
        #Boucle
        print "Cela vous convient-il ? [Oui / Non]:",
        resp = str(raw_input())
        if ((resp == "Oui") | (resp == "") | (resp == "O")):
            print "Fin."
            simu = False
            
def calcul_mensualite(emprunt, nb_mois, taeg, taux_ass, ammorti=True):
    """
    hypothese : Mensualite constante
    """
    #Calcul du taux mensuel
    taux_mens = []
    for i in range(1, nb_mois+1):
        taux_mens.append(1.0/math.pow((1+taeg), i/12.0))
    mens = emprunt/sum(taux_mens)
    somme_tot = mens*nb_mois
    #Calcul des interet et remboursement mensuel
    inte_mens = []
    remb_mens = []
    for i in range(1, nb_mois+1):
        remb = mens*taux_mens[-i] #Logique : la valeur de l'interet augmente avec la duree
        inte = mens - remb
        inte_mens.append(inte)
        remb_mens.append(remb)
    #Calcul mensualite assurance
    rapport = emprunt/10000
    mens_ass = taux_ass*10.0*rapport
    #Sortie Ecran
    print "Prix total --------------------------------------------------"
    print "Emprunt     :", somme_tot
    print "Emprunt+Ass.:", somme_tot + mens_ass*nb_mois
    print "Table de Mensualites ----------------------------------------"
    print "Mensualites :", mens
    print "Assurance   :", mens_ass
    print "Total       :", mens + mens_ass
    if ammorti:
        print "Repartition mois par mois :"
        print "-------------------------------------------------------------"
        print " Mois | Interet | Remboursement "
        print "-------------------------------------------------------------"
        for i in range(nb_mois):
            print " {0} | {1} | {2}".format(i+1, inte_mens[i], remb_mens[i]) 
        print "-------------------------------------------------------------"

def calcul_frais_notaire(prix, recent=False):
    """
    Hypothese : prix > 60000euros
    """
    #Droits de mutation (Fisc)
    taux_fisc = 5.80665/100.0
    fisc = prix*taux_fisc
    #Emoluments notarials (Remuneration)
    taux_nota = 0.814/100.0
    fixe_nota = 405.41
    nota = prix*taux_nota + fixe_nota
    #TVA Emoluments
    taux_tva = 20.0/100.0
    tva = nota*taux_tva
    #Emoluments de formalites
    form = 800.0
    #Frais dossier notaire
    frais = 400.0
    #Secu
    taux_secu = 0.10/100.0
    secu = max(15.0, prix*taux_secu)
    #Total
    total = fisc+nota+tva+form+frais+secu
    #Sortie Ecran
    print "Frais notariaux ---------------------------------------------"
    print "FISC                 : {0}".format(fisc)
    print "Remuneration         : {0}".format(nota)
    print "TVA sur remuneration : {0}".format(tva)
    print "Frais de formalites  : {0}".format(form)
    print "Frais divers         : {0}".format(frais)
    print "Contribution SECU    : {0}".format(secu)
    print "               TOTAL : {0}".format(total)
    #Sortie
    return total

if __name__=="__main__":
    print "Calculette de pret immobilier"
    while(1):
        print "Choisir un mode ============================================="
        print "     1 : Calcul de mensualites normales "
        print "     2 : Simulation d'achat de logement simple "
        print "     3 : Simulation d'achat de logement avec revente "
        print "Votre choix : ",
        usr_input = int(raw_input())
        if usr_input == 1:
            prompt_calcul_mensualite()
        elif usr_input == 2:
            prompt_simulation_achat()
        elif usr_input == 3:
            prompt_simulation_achat_revente()
        else:
            print "Erreur : mode inconnu"
