
# coding: utf-8

# In[1]:


import random
import math 
import pandas as pd

from mesa import Agent,Model
from mesa.time import RandomActivation


# In[2]:


'''
Model Agents involved : Payer, Manufacturer and Accounts

Role of manufacturer : Each manufacturer assigns 100 proposed price for each account but there is an inherent discount 
that is present from getting random values of discounts from "accounts_agents.csv" which serves as the mock data. 

Staging and Activation Procedure : At each step, 5 payers are being activated along with 1 manufacturer and 2 payers
(these numbers can be changed). A total of 100 mock payer agents are being activated in different stages.

Role of Accounts : An account has some discount capacity

Role of Payer : For each final active payer remaining, manufactures has to give 2 units discount on payer side 
and 1 unit discount 

Rule for Exchange : At each staged activation, the WAR (Weighted Average Rebate) is being calculated and 
the difference between account discount from .csv file and WAR matrix is also calculated, if that discount 
is positive, then the account remains in business otherwise the sum of discounted values is deducted from 
psp apart from the deductions of accounts commissions.

'''


class agent_abs(Model):
    
    def __init__(self,acc_file_loc,num_man = 1,num_payers = 2):
        
        self.acc_data  = pd.read_csv(acc_file_loc)
        self.num_man = num_man
        self.num_payers = num_payers
        self.num_accounts  = self.acc_data.shape[0]
        self.schedule  = RandomActivation(self)
       # self.DataCollector = ({"final_price_after_discount":final_pad})
        
    def batch_generator(self,df,col,batch_len = 5):
        batch = []
        overall_acc = []
        for i in range(0,len(df),batch_len):
            for j in range(i,i+batch_len):
                batch.append(df[col][j])
            overall_acc.append(batch)
            batch = []
        return overall_acc
    
            
    def run_agents(self):
        self.batch_len = 5
        self.id_batchwise = self.batch_generator(self.acc_data,'id',self.batch_len)
        self.rebate_batchwise = self.batch_generator(self.acc_data,'rebate',self.batch_len)
        self.money_spent = 0
        self.money_left = 0
        self.list_money_steps = []
        for batch,rebate in zip(self.id_batchwise,self.rebate_batchwise):
            
            # Agent creation of Accounts with other base configurations 
            self.rebate_total = 0
            for i in range(len(batch)):
                batch_acc = agent_accounts(batch[i],self,rebate[i])
                self.schedule.add(batch_acc)
                # WAR is Weighted Average Rebate
                self.rebate_total  = rebate[i]+ self.rebate_total
            self.WAR = self.rebate_total/len(batch)
            self.list_WAR  = [self.WAR]*len(batch)
            
            
            # Agent creation for Manufacturers
            for i in range(self.num_man):
                man = agent_manufacturer(i,self,psp = 100)
                self.schedule.add(man)
                
            # Agent creation for Payers
            for i in range(self.num_payers):
                payer =  agent_payer(i,self,mta_share=1,mtp_share=2)
                self.schedule.add(payer)
                
            self.money_spent = self.money_spent + 100*5
            self.list_difference = [] 
            
            for i in range(len(self.list_WAR)):
                self.list_difference.append(rebate[i]-self.list_WAR[i])
            
            self.sum_discounted = 0
            self.accounts_active_count = 0
            for i in range(len(self.list_difference)):
                if(self.list_difference[i]>0):
                    self.sum_discounted += self.list_difference[i]
                    self.accounts_active_count = self.accounts_active_count+1
            
            self.money_left  =  self.money_left + self.money_spent - self.sum_discounted - self.accounts_active_count*self.num_payers*2- self.accounts_active_count*self.num_man*1     
            self.list_money_steps.append(self.money_left)
        return self.list_money_steps


# In[3]:


# psp : proposed selling price
class agent_manufacturer(Agent):
    
    def __init__(self,unique_id,model,psp=100):
        super().__init__(unique_id,model)
        self.psp = psp


# In[4]:


class agent_payer(Agent):
    
    def __init__(self,unique_id,model,mta_share=1,mtp_share=2):
        super().__init__(unique_id,model)
        self.mta_share = mta_share
        self.mtp_share = mtp_share


# In[5]:


class agent_accounts(Agent):
    
    def __init__(self,unique_id,model,rebates):
        super().__init__(unique_id,model)
        self.rebate = rebates


# In[6]:


agent_abs = agent_abs('accounts_agents.csv',num_man=1,num_payers=2)


# In[7]:


a = agent_abs.run_agents()


# In[9]:


import matplotlib.pyplot as plt
plt.plot(a)
plt.show()

