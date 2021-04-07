from matplotlib import pyplot as plt
from autograd import numpy as np
from autograd import grad
import time 
import matplotlib.pyplot as plt
from utils import write_csv

# Question 11
def prediction_loss(x,y,W,V,b,c):
    """
    Compute loss with respect to the given forward propagation
    """
    l1_out    = np.dot(W,x)+b
    l1_act    = np.tanh(l1_out)
    final_out = np.dot(V,l1_act)+c
    L         = -final_out[y][0]+np.log(np.sum(np.exp(final_out))) # To return a scalar quantity
    return L
    
# Question 12 
def prediction_grad(x,y,W,V,b,c):
    """
    Function to compute gradients for 1 hidden layer neural network
    """
    l1_out    = np.dot(W,x)+b
    l1_act    = np.tanh(l1_out)
    final_out = np.dot(V,l1_act)+c
    
    ex_fv     = np.exp(final_out)
    ex_sum    = np.sum(np.exp(final_out))
    gt_calc   = ex_fv/ex_sum
    
    init_unit    = np.zeros(len(c))
    init_unit[y] = 1
    
    der_l1_act = 1 - np.square(l1_act) #Check this part
    
    dldf = -init_unit.reshape(len(c),1)+gt_calc
    dldc = dldf
    dldv = np.dot(dldf,l1_act.T)
        
    dldb = der_l1_act*(np.dot(V.T,dldf)) 
    
    dldw = np.outer(dldb,x.T)
    
    return dldw,dldv,dldb,dldc
    
# Question 14 
def prediction_grad_autograd(x,y,W,V,b,c):
    
    def prediction_loss_check(x,y,W,V,b,c):
        """
        Compute loss with respect to the given forward propagation
        """
        l1_out    = np.dot(W,x)+b
        l1_act    = np.tanh(l1_out)
        final_out = np.dot(V,l1_act)+c
        L         = -final_out[y][0]+np.log(np.sum(np.exp(final_out))) 
        return L
    
    g1 = grad(prediction_loss_check,5)
    dldc =  g1(x,y,W,V,b,c)
    g2 = grad(prediction_loss_check,4)
    dldb =  g2(x,y,W,V,b,c)
    g3 = grad(prediction_loss_check,3)
    dldV =  g3(x,y,W,V,b,c)
    g4=  grad(prediction_loss_check,2)
    dldw =  g4(x,y,W,V,b,c)
    
    return dldw,dldv,dldb,dldc
    
# Question 15
def prediction_loss_full(X,Y,W,V,b,c,l):
        """
        Function to calculate the prediction loss 
        for the given batch of data with regularization
        of weight matrices added
        Note that function assumes input of X in 2 dimensional 
        shape with first dimension containing all the examples in form of column vectors
        Y is a one dimensional input that needs to be transformed to get unit matrix
        """
    #     # Creating the unit matrix 
    #     b = np.zeros((Y.size, Y.max()+1))
    #     b[np.arange(Y.size),Y] = 1
    #     # Creating the ground-truth unit matrix for direct comparisons
    #     Y_gt = b.T

        # Working on the matrices and operations defined
        l1_out    = np.dot(W,X)+b
        l1_act    = np.tanh(l1_out)
        final_out = np.dot(V,l1_act)+c

        # Get losses accumulated 
        w2 = np.sum(np.square(W))
        v2 = np.sum(np.square(V))

        # Final loss calculations ahead
        log_val_axis = np.log(np.sum(np.exp(final_out),axis=0))
        sel_val_axis = final_out[Y,np.arange(len(Y))]
        # Overall loss with regularization part
        L = np.sum(-sel_val_axis+log_val_axis)+l*np.sum(w2+v2)
        return L

# Question 16
def prediction_grad_full(X,Y,W,V,b,c,l):
    
    def prediction_loss_full_check(X,Y,W,V,b,c,l):
        """
        Function to calculate the prediction loss 
        for the given batch of data with regularization
        of weight matrices added
        Note that function assumes input of X in 2 dimensional 
        shape with first dimension containing all the examples in form of column vectors
        Y is a one dimensional input that needs to be transformed to get unit matrix
        """

        # Working on the matrices and operations defined
        l1_out    = np.dot(W,X)+b
        l1_act    = np.tanh(l1_out)
        final_out = np.dot(V,l1_act)+c

        # Get losses accumulated 
        w2 = np.sum(np.square(W)) 
        v2 = np.sum(np.square(V))
        # Final loss calculations
        log_val_axis = np.log(np.sum(np.exp(final_out),axis=0))
        sel_val_axis = final_out[Y,np.arange(len(Y))]
        # Overall loss with regularization part
        L = np.sum(-sel_val_axis+log_val_axis)+l*np.sum(w2+v2)
        return L
    
   
    g1 = grad(prediction_loss_full_check,5)
    dldc =  g1(X,Y,W,V,b,c,l)
    g2 = grad(prediction_loss_full_check,4)
    dldb =  g2(X,Y,W,V,b,c,l)
    g3 = grad(prediction_loss_full_check,3)
    dldv =  g3(X,Y,W,V,b,c,l)
    g4=  grad(prediction_loss_full_check,2)
    dldw =  g4(X,Y,W,V,b,c,l)
    
    return dldw,dldv,dldb,dldc
    
# Question 17
def run_momentum_grad_descent(X,Y,M,momentum=0.1,iterations=1000,stepsize=0.001,l=1):
    """
    Code for running gradient descent with momentum on given data 
    """
    # Log time at the start of the training
    start_time = time.time()
    
    # Initialize weights and biases 
    b = np.zeros((M,1))
    c = np.zeros((len(set(Y)),1)) #Since there are 4 output classes only 
    W = np.random.randn(M,X.shape[1])/np.sqrt(X.shape[1]) #Input dimension is 3072
    V = np.random.randn(len(set(Y)),M)/np.sqrt(M) #Input dimension is M, output is number of classes i.e. 4
    
    # Momentum based gradient descent 
    ave_grad_w = 0
    ave_grad_b = 0
    ave_grad_v = 0
    ave_grad_c = 0
    
    loss_train = [] 
    
    # Run iterations for training
    for iter_no in range(0,iterations):
        
#         if ((iter_no%100)==0):
        print("Total iterations completed: ", iter_no)
            
        # Cost calculation using forward propagation
        L = prediction_loss_full(X.T,Y,W,V,b,c,l)
        loss_train.append(L)
#       print(L)
        # Gradient Calculation
        dldw,dldv,dldb,dldc = prediction_grad_full(X.T,Y,W,V,b,c,l)
        
        # Update first layer weights
        ave_grad_w = (1-momentum)*ave_grad_w + momentum * dldw
        W = W-stepsize*ave_grad_w
        
        # Update first layer biases
        ave_grad_b = (1-momentum)*ave_grad_b + momentum * dldb
        b = b-stepsize*ave_grad_b
        
        # Update second layer weights 
        ave_grad_v = (1-momentum)*ave_grad_v + momentum * dldv
        V = V-stepsize*ave_grad_v
        
        # Update second layer biases 
        ave_grad_c = (1-momentum)*ave_grad_c + momentum * dldc
        c = c-stepsize*ave_grad_c
        
    end_time = time.time()
    duration = end_time-start_time
        
    return loss_train,duration,W,b,V,c

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def make_predictions(X,W,b,V,c):
    "Expects input in form of transpose of X"
    l1_out    = np.dot(W,X)+b
    l1_act    = np.tanh(l1_out)
    final_out = np.dot(V,l1_act)+c
    pred      = [np.argmax(softmax(final_out[:,i])) for i in range(X.shape[1])]
    return pred

def create_train_val_split(X_trn,y_trn):
    
    idx_list = np.random.permutation(X_trn.shape[0])
    train_idx,val_idx = idx_list[:3000],idx_list[3000:]
    
    X_trn18, y_trn18  = X_trn[train_idx],y_trn[train_idx]
    X_val18, y_val18  = X_trn[val_idx], y_trn[val_idx]
    
    return X_trn18,y_trn18, X_val18, y_val18
    
def run_pipeline_momentum_gdesc(X_trn18, y_trn18, X_val18, y_val18, X_tst, M , momentum=0.1, iterations=1000, stepsize=0.0001, l=1, filename='predictions18_5.csv'):
    
    loss_train_18_5,duration_18_5,W_18_5,b_18_5,V_18_5,c_18_5 = run_momentum_grad_descent(X_trn18,y_trn18,M,momentum,iterations,stepsize,l)
    pred_18_5_val = np.array(make_predictions(X_val18.T,W_18_5,b_18_5,V_18_5,c_18_5))
    pred_18_5_tst = np.array(make_predictions(X_tst.T,W_18_5,b_18_5,V_18_5,c_18_5))
    accuracy_18_5_val = (sum(pred_18_5_val == y_val18))/len(y_val18)
    print("Accuracy corresponding to the selected net is: ", accuracy_18_5_val)
    write_csv(pred_18_5_tst,filename)
    return accuracy_18_5_val, loss_train_18_5,duration_18_5,W_18_5,b_18_5,V_18_5,c_18_5
    
def plot_loss_curve(loss_train_5, loss_train_40, loss_train_70):
    
    l_5       = loss_train_5
    l_40      = loss_train_40
    l_70      = loss_train_70
    x         = list(range(len(l_5)))
    
    plt.figure(figsize=(20,10))
    plt.grid()
    plt.title("Loss Profiles for three hidden layer configurations of 5, 40, 70")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss for the Network")
    plt.plot(x,l_5,label= "5 hidden layer")
    plt.plot(x,l_40,label= "40 hidden layer")
    plt.plot(x,l_70,label= "70 hidden layer")
    plt.legend()
    plt.plot()

