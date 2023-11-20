import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import random
import math 
import copy
import random
import argparse
import torch.optim as optim
import torch.nn as nn
import modeldefine
import numpy as np
from scipy.optimize import minimize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

alpha = 0.01

def UCB(A, phi):
    #### ucb term
    phi = phi.view(-1,1)
    try:
        tmp, LU = torch.linalg.solve(phi,A)
    except:
        A = A.detach().numpy()
        phi2 = phi.detach().numpy()
        tmp = torch.Tensor(np.linalg.solve(A, phi2))

    return torch.sqrt(torch.matmul(torch.transpose(phi,1,0), tmp))

def calculate_v(contextinfo_list, A, theta):
    vj_list = []
    feature_list = []
    for i in contextinfo_list:
        feature = model(i.to(device)).cpu()
        first_item =  torch.mm( feature.view(1,-1) , theta)
        second_item = alpha * UCB(A, feature)
        vj_list.append((first_item + second_item).item())
        feature_list.append(feature.detach().numpy())
    return np.array(vj_list), feature_list

def update_A(A, info_subset):
    for i in info_subset:
        i = torch.tensor(i, dtype=torch.float32,device=device)
        feature = model(i.to(device)).view(1,-1).cpu()
        A = A + torch.mm(feature.t(), feature)
    return A

# 这里调小utility 
def prob(vj_list):
    sum = np.sum(np.exp(vj_list)) + 1
    return [np.exp(vj_list[i]) / sum for i in range(len(vj_list))]  

def revenue(vj_list, reward_list):
    sum = np.sum(np.exp(vj_list)) + 1
    return np.sum(np.multiply(np.exp(vj_list), reward_list) / sum)

def assort(contextinfo_list, reward_list, vj_list, feature_list):
    length = len(contextinfo_list)
    # sort the contextinfo_list and vj with descending order of reward_list
    sorted_list = sorted(zip(contextinfo_list, vj_list, reward_list, feature_list), key=lambda x: x[2], reverse=True)
    
    contextinfo_list = [x[0] for x in sorted_list]
    vj_list = [x[1] for x in sorted_list]
    reward_list = [x[2] for x in sorted_list]
    feature_list = [x[3] for x in sorted_list]

    # calculate the optimal assortment
    optimal_assort = []
    optimal_reward = revenue(vj_list[:1], reward_list[:1])
    index = 1 
    for i in range(2,length):
        if revenue(vj_list[:i], reward_list[:i]) >= optimal_reward:
            optimal_reward = revenue(vj_list[:i], reward_list[:i])
        else:
            index = i - 1 
            break
    return contextinfo_list[:index], feature_list[:index], vj_list[:index], reward_list[:index]

# this is for the non_linear purchase model when v = x dot theta
def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def get_nonlinear_purchase(feature_list):
    true_Vlist = [(TRUE_THETA @ sigmoid(feature_list[i]).reshape(-1,1)).item() for  i in range(len(feature_list))]
    prob_list = prob(true_Vlist)

    # sample item according to prob_list
    if random.uniform(0,1) < 1 - np.sum(prob_list):
        return np.array([0 for i in range(len(feature_list))])
    else:
        returnlist = [0 for i in range(len(feature_list))]
        indexchoose = random.choices([i for i in range(len(prob_list))], weights = prob_list)[0]
        returnlist[indexchoose] = 1
        return np.array(returnlist)
    

lambd = 0.1
def likelihood(theta, feature_list ,y_list):
    # feature's dimension is len * dimension , theta is 1*dimension
    v_list = np.matmul(feature_list, theta.T).reshape(-1)
    ln_prob = np.log(prob(v_list))
    summation = ln_prob * y_list
    return -1 * np.sum(summation)

def likelihood_derivative(theta, feature_list, y_list):
    v_list = np.matmul(feature_list, theta.T).reshape(-1)
    prob_list = prob(v_list)
    summation = np.matmul(np.array(feature_list).T, (y_list - prob_list))
    return -1 * summation

def likelihood_array(theta, feature_list_list, y_list_list):
    summation =  0.5 * lambd * np.dot(theta, theta)
    for i in range(len(feature_list_list)):
        summation += likelihood(theta, feature_list_list[i], y_list_list[i])
    return summation

def likelihood_derivative_array(theta, feature_list_list, y_list_list):
    summation = 0.5 * lambd * theta
    for i in range(len(feature_list_list)):
        summation += likelihood_derivative(theta, feature_list_list[i], y_list_list[i])
    return summation


class CustomLikelihoodLoss(nn.Module): 
    def __init__(self, theta_list):
        super(CustomLikelihoodLoss, self).__init__()
        self.theta_list = theta_list

    def forward(self, output_list, y_list):
        loss = 0
        index = 0
        for output in output_list:  
            y = torch.tensor(y_list[index]).to(device).view(-1,1)
            theta = torch.tensor(self.theta_list[index], dtype= torch.float32).to(device) 
            v = torch.mm(output, theta.view(-1,1)) 
            prob = torch.exp(v) / (torch.sum(torch.exp(v)) + 1)  
            loss += torch.sum(torch.log(prob) * y)  
            index += 1  
        return -loss 

class CustomLikelihoodLoss2(nn.Module):
    def __init__(self, theta_list):
        super(CustomLikelihoodLoss2, self).__init__()
        self.theta_list = theta_list
     
    def forward(self, output_list, y_list):
        loss = 0
        index = 0
        for output in output_list:
            y = torch.tensor(y_list[index]).to(device).view(-1,1)
            theta = torch.tensor(self.theta_list[index], dtype= torch.float32).to(device)
            v = torch.mm(output, theta.view(-1,1))
            prob = torch.exp(v) / (torch.sum(torch.exp(v)) + 1)
            # ce loss between prob and y
            loss += torch.sum(-y * torch.log(prob) - (1-y) * torch.log(1-prob))
            index += 1
        return 100*loss/len(output_list) 

# this block is only used for linear model revenue calculation
def calculate_sigmoid_v(contextinfo_list, A, theta):
    vj_list = []
    feature_list = []
    for i in contextinfo_list:
        feature = sigmoid(i)
        first_item =  torch.mm( feature.view(1,-1) , theta)

        vj_list.append(first_item.item())
        feature_list.append(feature.detach().numpy())
    return np.array(vj_list), feature_list

# 真实情况乱下的theta，feature
def get_true_sigmoid_ass(context, profit):
    
    v_array = np.array(sigmoid(context) @ TRUE_THETA.T ).reshape(-1)
    assortment, ass_features, v_list, profit_list = assort(context, profit.tolist()[0], v_array.tolist() , sigmoid(context))
    true_probablility = np.array(prob(v_list))
    revenue = np.dot(true_probablility, np.array(profit_list))
    return revenue


def get_assortment_revenue(assortment, profit):
    v_array = np.array(sigmoid(np.array(assortment)) @ TRUE_THETA.T ).reshape(-1)
    true_probablility = np.array(prob(v_array))
    revenue = np.dot(true_probablility, np.array(profit))
    return revenue


import modeldefine
import torch.optim as optim
model = modeldefine.Model(5,10,10,4).to(device)
# 10 20 20 20 20 20  5
optimizer = optim.Adam(model.parameters(), lr=0.001)

# data reader
CONTEXT_ARRAY = np.load('nonlinear_data/features.npy') 
REWARD_ARRAY = np.load('nonlinear_data/rewards.npy')
TRUE_THETA = np.load('nonlinear_data/theta.npy')


data_length = len(CONTEXT_ARRAY)

# define the hyperparameters
input_size = 20
hidden_size = 20
output_size = 10
num_layers = 10

beta = 0.1

H = 100

# initialize the parameters

theta = np.random.randn(output_size) / np.sqrt(output_size)
#theta = TRUE_THETA
LAMBDA = lambd * torch.eye(output_size, dtype=torch.float32)

ass_list = []
feature_list = []
purchase_list = []
theta_list = []

revenue_list1 = []
revenue_list2 = []
true_profit_list = []

for t in range(0, 10000):
    context = CONTEXT_ARRAY[t]
    profit = REWARD_ARRAY[t]

    theta_tensor = torch.tensor(theta.reshape(-1,1), dtype=torch.float32)
    v_array,initial_feature = calculate_v(torch.tensor(context,dtype=torch.float32), LAMBDA, theta_tensor)
    assortment, ass_features, vv_list , reward = assort(context, profit.tolist()[0], v_array.tolist() , initial_feature)
    purchase_vector = get_nonlinear_purchase(assortment)
    
    # calculate the ideal 
    expected_revenue1 = get_true_sigmoid_ass(context, profit)
    revenue_list1.append(expected_revenue1)
    true_profit_list.append(np.dot(np.array(purchase_vector), reward))
    revenue_list2.append(get_assortment_revenue(assortment, reward))

    # add to list
    ass_list.append(np.array(assortment))
    feature_list.append(np.array(ass_features))
    purchase_list.append(purchase_vector)

    # update the parameters
    LAMBDA = update_A(LAMBDA, assortment)
    
    # update theta using MLE
    initial_guess = theta
    
    try:
        result = minimize(likelihood_array, initial_guess, args=(feature_list, purchase_list), method='SLSQP', 
                  constraints={'type':'eq', 'fun': likelihood_derivative_array, 'args':(feature_list, purchase_list)})
        theta = result.x
    except: 
        print('error occured')
        theta = theta
    theta_list.append(theta)
    
    #theta_list.append(TRUE_THETA)
    # update the neural networks
    if t % H == 99:
        #a_list = ass_list[-1*H:]
        #y_list = purchase_list[-1*H:]

        a_list = ass_list[:]
        y_list = purchase_list[:]

        loss_function = CustomLikelihoodLoss2(theta_list)
        epochs = 3
     
        for epoch in range(epochs):
            output_list = [model(torch.tensor(a,dtype = torch.float32).to(device)) for a in a_list]
            loss = loss_function(output_list, y_list)
            #if (epoch == epochs-1): 
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #theta_list = []


np.save('nonlinear_data/record/1mtrue_revenue_list1.npy', np.array(revenue_list1))
np.save('nonlinear_data/record/1msimulate_revenue_list2.npy', np.array(revenue_list2))
np.save('nonlinear_data/record/1mpurchase_revenue_list.npy', np.array(true_profit_list))