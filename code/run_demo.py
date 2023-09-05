import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import math
import copy
import random
import argparse
import datetime
print(datetime.datetime.now().strftime("%d%B%Y, %H:%M:%S"))
device = torch.device('cpu')

# data loader
# 返回对象维context_size, arm_size, context, reward, sli(排序)
def load_data(dataset_name, K):
    if dataset_name == "statlog":
        ##statlog
        __context_size__ = 8
        __arm_size__ = 7
        fr = open("shuttle.trn",'r+')
        CONTEXT = []
        REWARD = []
        SYM = {}
        aa = 0
        K = 0
        for line in fr:
            K+=1
            aaa = line.split("\n")
            aaa = aaa[0].split(" ")
            temp = np.int64(aaa[9])-1
            context = np.double(aaa[1:9])
            context = context/np.linalg.norm(context)
            ttt = context
            CONTEXT.append(ttt)
            reward = np.zeros(__arm_size__)
            reward[temp] = 1
            REWARD.append(reward)

    elif dataset_name == "magic":
        ####magic gamma

        __context_size__ = 10
        __arm_size__ = 2
        fr = open("letter.data", 'r+')

        K = 0
        CONTEXT = []
        REWARD = []

        fr.close()
        fr = open("magic04.data", 'r+')
        for line in fr:

            context = []
            aaa = line.split(",")
            context = np.double(aaa[0:__context_size__])
            context = context / np.linalg.norm(context)
            K += 1
            CONTEXT.append(np.array(context))
            __context_size__ = len(context)
            reward = np.zeros(__arm_size__)
            if aaa[10] == 'g\n':
                reward[0] = 1
            else:
                reward[1] = 1
            REWARD.append(reward)

        fr.close()
    elif dataset_name == "covertype":
        ##For covertypr
        __context_size__ = 14
        __arm_size__ = 7
        fr = open("covtype.data", 'r+')
        CONTEXT = []
        REWARD = []
        SYM = {}
        aa = 0
        K = 0
        for line in fr:
            K += 1
            aaa = line.split(",")
            __context_size__ = len(aaa) - 1
            temp = np.int(aaa[__context_size__]) - 1
            context = np.double(aaa[0:__context_size__])
            context = context / np.linalg.norm(context)
            ttt = context
            CONTEXT.append(ttt)
            reward = np.zeros(__arm_size__)
            reward[temp] = 1
            REWARD.append(reward)
    else:
        assert('dataset not avaiable')

    TEMP_CONTEXT = []
    TEMP_REWARD = []
    sli = np.random.permutation(K)
    for i in range(K):
        TEMP_CONTEXT.append(CONTEXT[sli[i]])
        TEMP_REWARD.append(REWARD[sli[i]])
    CONTEXT = TEMP_CONTEXT
    REWARD = TEMP_REWARD

    return __context_size__, __arm_size__, CONTEXT, REWARD, sli
 
# Initialization，利用kronecker product初始化参数，详见p5
def INI(dim):  
    #### initialization
    #### dim consists of (d1, d2,...), where dl = 1 (placeholder, deprecated)
    w = []
    total_dim = 0
    for i in range(0, len(dim) - 1):
        if i < len(dim) - 2:
            temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i + 1])
            temp = np.kron(np.eye(2, dtype=int), temp)
            temp = torch.from_numpy(temp).to(device)
            w.append(temp)
            total_dim += dim[i + 1] * dim[i] *4
        else:
            temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i])
            temp = np.kron([[1, -1]], temp)
            temp = torch.from_numpy(temp).to(device)
            w.append(temp)
            total_dim += dim[i + 1] * dim[i]*2

    return w, total_dim

# 模拟神经网络的输出
def FUNC_FE(x, W):  
    #### Functions feature extractor
    #### x is the input, dimension is d; W is the list of parameter matrices
    depth = len(W)
    output = x
    for i in range(0, depth - 1):
        output = torch.mm(W[i], output)
        output = output.clamp(min=0) # relu激活函数

    output = output * math.sqrt(W[depth - 1].size()[1])
    return output

def GRAD_LOSS(X, Y, W, THETA):
    ##return a list of grad, satisfying that W[i] = W[i] - grad[i] ##for single context x
    depth = len(W)
    num_sample = Y.shape[0]
    loss = []
    grad = []
    relu = []
    output = X
    loss.append(output)
    for i in range(0, depth - 1):
        output = torch.mm(W[i], output)
        relu.append(output)
        output = output.clamp(min=0)
        loss.append(output)

    THETA_t = torch.transpose(THETA,0,1).view(num_sample, 1, -1)
    output_t = torch.transpose(output,0,1).view(num_sample, -1, 1)
    output = torch.bmm(THETA_t, output_t).squeeze().view(1,-1)      

    loss.append(output)
    ####
    feat = FUNC_FE(X, W)
    feat_t = torch.transpose(feat, 0, 1).view(num_sample, -1, 1)
    output_t = torch.bmm(THETA_t, feat_t).squeeze().view(1, -1)

    #### backward gradient propagation
    back = output_t - Y
    back = back.double()
    grad_t = torch.mm(back, loss[depth - 1].t())
    grad.append(grad_t)

    for i in range(1, depth):
        back = torch.mm(W[depth - i].t(), back)
        back[relu[depth - i - 1] < 0] = 0
        grad_t = torch.mm(back, loss[depth - i - 1].t())
        grad.append(grad_t)
    #### 
    grad1 = []
    for i in range(0, depth):
        grad1.append(grad[depth - 1 - i] * math.sqrt(W[depth - 1].size()[1]) / len(X[0, :]))

    if (grad1[0] != grad1[0]).any():
        print('nan found')
        import sys; sys.exit('nan found') 
    return grad1

def loss(X, Y, W, THETA):
    #### total loss
    num_sample = len(X[0, :])
    output = FUNC_FE(X, W)
    THETA_t = torch.transpose(THETA, 0, 1).view(num_sample, 1, -1)
    output_t = torch.transpose(output, 0, 1).view(num_sample, -1, 1)
    output_y = torch.bmm(THETA_t, output_t).squeeze().view(1, -1)

    summ = (Y - output_y).pow(2).sum() / num_sample
    return summ

def TRAIN_SE(X, Y, W_start, T, et, THETA, H):  
    #### gd-based model training with shallow exploration
    #### dataset X, label Y
    W = copy.deepcopy(W_start)
    num_sample = H
    X = X[:, -H:]
    Y = Y[-H:]
    THETA = THETA[:, -H:]

    prev_loss = 1000000
    prev_loss_1k = 1000000
    for i in range(0, T):
        grad = GRAD_LOSS(X, Y, W, THETA)
        if (grad[0] != grad[0]).any():
            print('nan found')
        for j in range(0, len(W)-1):
            W[j] = W[j] - et * grad[j]

        curr_loss = loss(X, Y, W, THETA)
        if i % 100 == 0:
            print('------',curr_loss)
            if curr_loss > prev_loss_1k:
                et = et * 0.1
                print('lr/10 to', et)

            prev_loss_1k = curr_loss

        # early stopping
        if abs(curr_loss - prev_loss) < 1e-6:
            break
        prev_loss = curr_loss
    return W

def UCB(A, phi):
    #### ucb term
    try:
        tmp, LU = torch.linalg.solve(phi,A)
    except:
        tmp = torch.Tensor(np.linalg.solve(A, phi))

    return torch.sqrt(torch.mm(torch.transpose(phi,0,1), tmp.double()))

def TRANS(c, a, arm_size):
    #### transfer an array context + action to new context with dimension 2*(__context__ + __armsize__)
    dim = len(c)
    action = np.zeros(arm_size)
    action[a] = 1
    c_final = np.append(c, action)
    c_final = torch.from_numpy(c_final).to(device)
    c_final = c_final.view((len(c_final), 1))
    c_final = c_final.repeat(2, 1)
    return c_final

def main(args):
    #### main function
    # load data
    __context_size__, __arm_size__, CONTEXT, REWARD, sli = load_data(args.dataset, args.K)
    # paras initialization
    lambd = args.lambd
    hid_dim_lst = args.hidden_dim
    dim_second_last = args.hidden_dim[-1] *2

    dim_for_init = [__context_size__ + __arm_size__] + hid_dim_lst + [1]
    W0, total_dim = INI(dim_for_init)
    LAMBDA = lambd * torch.eye(dim_second_last, device=device, dtype=torch.double)
    bb = torch.zeros(LAMBDA.size()[0], device=device, dtype=torch.double).view(-1, 1)

    theta = np.random.randn(dim_second_last, 1) / np.sqrt(dim_second_last)
    theta = torch.from_numpy(theta).to(device)

    THETA_action = []
    CONTEXT_action = []
    REWARD_action = []
    result_neuralucb = []
    W = copy.deepcopy(W0)
    summ = 0

    for t in range(0, args.K):
        # first, calculate estimated value for different actions
        context = CONTEXT[t]
        ucb = []
        bphi = []
        for a in range(0, __arm_size__):
            temp = TRANS(context, a, __arm_size__)
            bphi.append(temp)
            feat = FUNC_FE(temp, W)
            ucb.append(torch.mm(theta.view(1,-1), feat) + args.beta * UCB(LAMBDA, feat))

        # second, choose an action
        # use round-robin, #initial_pull = 3
        if t< 3*__arm_size__:
            a_choose = t % __arm_size__
        else:
            a_choose = ucb.index(max(ucb))

        # third, get reward
        reward = REWARD[t][a_choose]
        summ += (max(REWARD[t]) - reward)
        result_neuralucb.append(summ)

        # finally update W by doing TRAIN_SE
        if np.mod(t, args.H_q) == 0:
            CONTEXT_action = []
            REWARD_action = []
            CONTEXT_action = bphi[a_choose]
            REWARD_action = torch.tensor([reward], device=device, dtype=torch.double)
        else:
            CONTEXT_action = torch.cat((CONTEXT_action, bphi[a_choose]), 1)
            REWARD_action = torch.cat((REWARD_action, torch.tensor([reward], device=device, dtype=torch.double)), 0)
        
        # update LAMBDA and bb
        LAMBDA += torch.mm(FUNC_FE(bphi[a_choose], W), FUNC_FE(bphi[a_choose], W).t())
        bb += reward * FUNC_FE(bphi[a_choose], W)
        #theta, LU = torch.linalg.solve(bb,LAMBDA)
        theta = torch.linalg.solve(LAMBDA, bb)

        if np.mod(t, args.H_q) == 0:
            THETA_action = []
            THETA_action = theta.view(-1,1)
        else:
            THETA_action = torch.cat((THETA_action, theta.view(-1,1)), 1)

        if np.mod(t, args.H_q) == args.H_q-1:
            print(summ)
            W = TRAIN_SE(CONTEXT_action, REWARD_action, W0, args.interT, args.et, THETA_action, args.H_q)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lambd', type=float, help='lambd', default=1)
    argparser.add_argument('--et', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--beta', type=float, help='parameter for UCB exploration', default=0.02)
    argparser.add_argument('-H_q', type=int, help='how many time steps to update NN', default=100)
    argparser.add_argument('--K', type=int, help='epoch number', default=15000)
    argparser.add_argument('--interT', type=int, help='internal steps for GD', default=1000)
    argparser.add_argument('--hidden_dim', type=int, nargs='+', help='hidden dim', default=[1000, 1000])
    argparser.add_argument('--filePath', type=str, help='result saving directory', default='./result_figs/')
    argparser.add_argument('--dataset', help='dataset name: statlog/magic/covertype', default="statlog")

    args = argparser.parse_args()
    print(args)

    if not os.path.exists(args.filePath):
        os.makedirs(args.filePath)

    main(args)

    print('done')

