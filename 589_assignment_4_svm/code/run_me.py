from utils import *
import numpy as np
from assgn4_functions import *
import matplotlib.pyplot as plt




# Load data 
print("Loading data for operations for further questions")
print("**************************************************************************")
stuff = np.load("data_synth.npz")
X_trn = stuff['X_trn']
Y_trn = stuff['Y_train']
X_val = stuff['X_val']
Y_val = stuff['Y_val']

# Train and Plot Basis Expansion for Q7 
print("Printing weight matrices and plotting curves for different values of P")
train_plot_basis_expansion_q7(X_trn,Y_trn,P=[1,2,3,5,10],λ = 0.1,ulim=15)
print("***************************************************************************")

# Performing operations for question 9
x  = 0.5
xp = 0.7
k  = get_poly_kernel(5)
h  = get_poly_expansion(5)
out1 = k(x,xp)
out2 = np.inner(h(x),h(xp))
print("output 1", out1)
print("output 2", out2)
print("***************************************************************************")

# Plotting results for Q12 
print("Plotting results for Question 12")
plot_kernel_ridge_results(X_trn,Y_trn,P = [1,2,3,5,10],λ = 0.1 ,ulim=15)
print("***************************************************************************")

#Load other data 
print("Loading other data")
stuff=np.load("data_real.npz")
x_trn = stuff["x_trn"]
y_trn = stuff["y_trn"]
x_tst = stuff["x_tst"]
print("***************************************************************************")

# Running sklearn related functions

print("Printing results for question 14")
run_question_14_svm(x_trn,y_trn,c = [2,20,200])
print("***************************************************************************")
 
print("Printing results for question 15")
run_question_15_svm(x_trn,y_trn, c = [2,20,200], gamma = [1,0.01,0.001],degree=3)
print("***************************************************************************")

print("Printing results for question 16")
run_question_16_svm(x_trn,y_trn, c = [2,20,200], gamma = [1,0.01,0.001],degree=5)
print("***************************************************************************")

print("Printing results for question 17")
run_question_17_svm(x_trn,y_trn,c = [2,20,200],gamma = [1,0.01,0.001],kernel='rbf')
print("***************************************************************************")

print("Running question 18")
y_pred = run_question_18_svm(x_trn,y_trn,x_tst,kernel='rbf', C=1/2, gamma=0.01)
print("***************************************************************************")



