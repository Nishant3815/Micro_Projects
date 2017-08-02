
# coding: utf-8

# In[1]:

# Initial imports
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

import os
import collections
import tensorflow as tf
from tensorflow.contrib import rnn
import random


# In[2]:

random.seed(3)


# In[3]:

# Imports for better visualization
from matplotlib import rcParams
#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (8, 3)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#eeeeee'


# In[ ]:




# In[4]:

# Loading Data
data = pd.read_csv("Data/train.csv")
evaluation = pd.read_csv("Data/test.csv")
sample_submission = pd.read_csv("Data/sample_submission.csv")


# In[5]:

unique_values = pd.DataFrame(data.columns, columns=["Column"])
unique_values["Type"] = unique_values["Column"].apply(lambda x: data[x].dtype)
unique_values["UniqueValues"] = unique_values["Column"].apply(lambda x: data[x].nunique())
unique_values


# In[6]:

data.Date.unique()


# In[7]:

# Sorting
data = data.sort(['PID', 'Date'], ascending=[True, True])
data = data.reset_index(drop=True)

# PID wise arrangement
data = data.groupby('PID').agg(lambda x: x.tolist())
data.reset_index(level=0, inplace=True)
del data["Date"]


# In[8]:

data.head()


# In[ ]:




# In[9]:

# Training data 
training_data = np.array(data["Event"].iloc[0])


# In[10]:

# Making Dictionary
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)


# In[11]:

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 5

n_hidden = 512


# In[12]:

# Placeholder variables
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])


# In[13]:

# RNN output node weights and biases
weights = {'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))}
biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}


# In[14]:

def RNN(x, weights, biases):

    # Reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    x = tf.split(x,n_input,1)

    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Using RMSProp optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[ ]:

with tf.Session() as session:
    
    # Initializing all variables
    session.run(tf.global_variables_initializer())
    
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    checkpoint = 0
    
    while step < training_iters:
        
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred],                                                 feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " +                   "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " +                   "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += (n_input+1)
        checkpoint_name = os.path.join("Checkpoints/", 'model_epoch'+str(checkpoint)+'.ckpt')
        save_path = tf.train.Saver().save(session, checkpoint_name)
        checkpoint +=1 


# In[ ]:




# In[19]:

testing_data = np.array(data["Event"].iloc[0][-30:-15])


# In[20]:

testing_data


# In[21]:

# Using first ten probability for prediction
with tf.Session() as sess:
    
    # Restore the pretrained weights
    tf.train.Saver().restore(sess, "Checkpoints/model_epoch30.ckpt")
    
    symbols_in_keys = [ [dictionary[ str(testing_data[i])]] for i in range(0, 5) ]
    symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

    onehot_pred = sess.run( pred, feed_dict={x: symbols_in_keys})
    symbols_in = [testing_data[i] for i in range(0, 5)]
    list_pred = onehot_pred
    indexes_list_pred = sorted(range(len(list_pred[0])), key=lambda i: list_pred[0][i])[-10:]
    symbols_out_pred= [reverse_dictionary[p] for p in indexes_list_pred]
#     symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
    print("%s - [%s]" % (symbols_in,symbols_out_pred))


# In[ ]:




# In[1]:

#no

