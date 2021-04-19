import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import hinge_loss
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import hinge_loss
import matplotlib.pyplot as plt
from utils import *

def eval_basis_expanded_ridge(x,w,h):
    expansion = h(x)
    y = np.dot(w,expansion.T)
    return y

def train_basis_expanded_ridge(X,Y,λ,h = get_poly_expansion(3)):
    H = h(X)
    C = np.matmul(H.T,H) + λ*np.identity(H.shape[1])
    w = np.linalg.solve(C,np.matmul(H.T,Y))
    return w
    
    
def train_plot_basis_expansion_q7(X_trn,Y_trn,P=[1,2,3,5,10],λ = 0.1,ulim=15):
    
    for p in P:
        h = get_poly_expansion(p)
        w = train_basis_expanded_ridge(X_trn,Y_trn,λ,h)
        print("Printing the weight vectors for different P in Q7")
        print("Weight vector = " + str(w) + " for P = " + str(p) + "\n")
        print("####################################")
    
    for p in P:
        h = get_poly_expansion(p)
        w = train_basis_expanded_ridge(X_trn,Y_trn,λ,h)
        x = np.linspace(0,ulim,1000,endpoint = True)
        fwx = eval_basis_expanded_ridge(x,w,h)
        text =  "Learned Function Output for P= "+str(p)
        plt.scatter(x,fwx,s=ulim,label = text)
        plt.scatter(X_trn,Y_trn,s =ulim,label = "Training Data")
        plt.legend()
        plt.show()
    
def get_poly_kernel(P):
    def k(x,xp):
        kernel_value = (1 + np.inner(x, xp)) ** P
        return kernel_value
    return k
    
def train_kernel_ridge(X,Y,λ,k):
    # Initiate a Kernel matrix
    K = np.zeros((X.shape[0],X.shape[0]))
    # Fill values for kernel matrix using kernel function
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            K[i][j] = k(X[i],X[j])
    # Get first part for np.linalg.solve
    kli = K+λ*np.identity(K.shape[0])
    # Compute alpha
    α   = np.linalg.solve(kli,Y)
    return α

def eval_kernel_ridge(X_trn, x, α, k):
    # Evaluation of kernel ridge regression
    sum_all = 0
    for i in range(len(X_trn)):
        sum_all += α[i]*k(X_trn[i],x)
    y = sum_all
    return y

def plot_kernel_ridge_results(X_trn,Y_trn,P = [1,2,3,5,10],λ = 0.1 ,ulim=15):
    
    for p in P:
        k = get_poly_kernel(p)
        α = train_kernel_ridge(X_trn,Y_trn,λ,k = get_poly_kernel(p))
        x = np.linspace(0,ulim,100)
        Y = []
        for i in range(len(x)):
            Y.append(eval_kernel_ridge(X_trn, x[i], α, k))
        Y = np.array(Y)
        text =  "Learned Function Output for P= "+str(p)
        plt.scatter(x,Y,s=ulim,label = text)
        plt.scatter(X_trn,Y_trn,s =ulim,label = "Training Data")
        plt.legend()
        plt.show()


def run_question_14_svm(x_trn,y_trn,c = [2,20,200]):
    
    for penalty in c:
        hinge_losses = []

        kf = KFold(n_splits=5, shuffle=True, random_state=3815)
        for train_index, test_index in kf.split(x_trn):


            x_trn_5, x_tst_5 = x_trn[train_index], x_trn[test_index]
            y_trn_5, y_tst_5 = y_trn[train_index], y_trn[test_index]


            clf = svm.SVC(kernel = 'linear', C = 1/penalty)
            clf.fit(x_trn_5,y_trn_5)

            y_pred = clf.decision_function(x_tst_5)
            hinge_losses.append(hinge_loss(y_tst_5,y_pred))



        mean_hinge_loss = (sum(hinge_losses)/5)
        print("The mean hinge loss with 5-Fold cross validation using hinge loss with lambda = " + str(penalty) + " is: ", mean_hinge_loss)


def run_question_15_svm(x_trn,y_trn, c = [2,20,200], gamma = [1,0.01,0.001],degree=3):
    
    for penalty in c:
          for g in gamma:

            hinge_losses = []

            kf = KFold(n_splits=5, shuffle=True, random_state=3815)
            for train_index, test_index in kf.split(x_trn):

                x_trn_5, x_tst_5 = x_trn[train_index], x_trn[test_index]
                y_trn_5, y_tst_5 = y_trn[train_index], y_trn[test_index]

                clf = svm.SVC(kernel = 'poly', degree = degree, C = 1/penalty, gamma = 1, coef0 = g)
                clf.fit(x_trn_5,y_trn_5)
                y_pred = clf.decision_function(x_tst_5)
                hinge_losses.append(hinge_loss(y_tst_5,y_pred))
            mean_hinge_loss = (sum(hinge_losses)/5)
            print("The mean hinge loss with 5-Fold cross validation using hinge loss with lambda = " + str(penalty)  + " and gamma = " + str(g) + " is: " , mean_hinge_loss)

def run_question_16_svm(x_trn,y_trn, c = [2,20,200], gamma = [1,0.01,0.001],degree=5):
    
    for penalty in c:
          for g in gamma:

            hinge_losses = []

            kf = KFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(x_trn):

                x_trn_5, x_tst_5 = x_trn[train_index], x_trn[test_index]
                y_trn_5, y_tst_5 = y_trn[train_index], y_trn[test_index]

                clf = svm.SVC(kernel = 'poly', degree = degree, C = 1/penalty, gamma = 1, coef0 = g)
                clf.fit(x_trn_5,y_trn_5)
                y_pred = clf.decision_function(x_tst_5)
                hinge_losses.append(hinge_loss(y_tst_5,y_pred))
            mean_hinge_loss = (sum(hinge_losses)/5)
            print("The mean hinge loss with 5-Fold cross validation using hinge loss with lambda = " + str(penalty)  + " and gamma = " + str(g) + " is: " , mean_hinge_loss)


def run_question_17_svm(x_trn,y_trn,c = [2,20,200],gamma = [1,0.01,0.001],kernel='rbf'):
    
    for penalty in c:
        for g in gamma:
            hinge_losses = []

            kf = KFold(n_splits=5, shuffle=True, random_state=3815)
            for train_index, test_index in kf.split(x_trn):

                x_trn_5, x_tst_5 = x_trn[train_index], x_trn[test_index]
                y_trn_5, y_tst_5 = y_trn[train_index], y_trn[test_index]

                clf = svm.SVC(kernel = kernel, C = 1/penalty, gamma = g)
                clf.fit(x_trn_5,y_trn_5)
                y_pred = clf.decision_function(x_tst_5)
                hinge_losses.append(hinge_loss(y_tst_5,y_pred))
            mean_hinge_loss = (sum(hinge_losses)/5)
            print("The mean hinge loss with 5-Fold cross validation using hinge loss with lambda = " + str(penalty)  + " and gamma = " + str(g) + " is: " , mean_hinge_loss)
            

def run_question_18_svm(x_trn,y_trn,x_tst,kernel='rbf', C=1/2, gamma=0.01):
    
    clf_final = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf_final.fit(x_trn,y_trn)
    y_pred = clf_final.predict(x_tst)
    y_pred_li = np.array([int(i) for i in y_pred])
    
    return y_pred_li

