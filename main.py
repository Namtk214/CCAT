# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict, deque
import json
import scipy
import copy as cp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import defaultdict
from dataset import Dataset, TrainDataset, AdapTestDataset
from setting import *
from IRT import IRTmodel

#IRT method
def IRT(x,a,b):
    return 1/(1+np.exp(-a*(x-b)))

def Likelihood(x,y,a,b):
    return (a*(y-1/(1+np.exp(-a*(x-b))))).sum()

def P(x,a,b):
    return 1/(1+torch.exp(-a*(x-b)))

#intra Ranking consistency
def Rank_score(train_label,test_label,selected_questions,theta_star,train_theta_star,ga,device):
    train_star = torch.FloatTensor(train_theta_star).to(device)
    pair=[0 for i in range(20)]
    for stu in range(len(test_label)):
        S=ga[selected_questions[stu]]*(train_label[:,selected_questions[stu]]-test_label[stu,selected_questions[stu]])
        for t in range(20):
            pair[t] += ((S[:,:t+1].sum(-1)*(train_star-theta_star[stu]))>=0).sum()
    return [i.item()/len(train_label)/len(test_label) for i in pair] 
    
def get_rank_result(train_label,test_label,selected_questions,ga,device):
    train_s=[[0 for i in range(20)] for j in range(len(test_label))]
    for stu in range(len(test_label)):
        S=ga[selected_questions[stu]]*(train_label[:,selected_questions[stu]]-test_label[stu,selected_questions[stu]])
        for t in range(20):
            train_s[stu][t]=(1-torch.sigmoid(S[:,:t+1].sum(-1))).cpu().numpy()
    return np.array(train_s)
#inter Ranking consistency
def pairwise(stu_theta,theta_star):
    pair=[]
    for t in range(20):
        sum=0
        for i in range(len(stu_theta)):
            for j in range(len(stu_theta)):
                if (stu_theta[i][t]-stu_theta[j][t])*(theta_star[i]-theta_star[j])>0:
                    sum+=1
        pair.append(sum/(len(stu_theta)*(len(stu_theta)-1)))  
    return pair    
#Acc score
def ACC_score(stu_theta,test_select,gamma,beta,test_label):
    acc = []
    for t in range(20):
        acc_sum = 0
        sum = 0
        for stu in range(len(stu_theta)):
            for q in test_select[stu]:
                acc_sum+=(round(IRT(stu_theta[stu][t],gamma[q],beta[q]))==test_label[stu][q]).item()
                sum+=1
        acc.append(acc_sum/sum)
    return acc
def AUC_score(stu_theta,test_select,gamma,beta,test_label):
    auc = []
    for t in range(20):
        pred = []
        true = []
        for stu in range(len(stu_theta)):
            for q in test_select[stu]:
                pred.append(IRT(stu_theta[stu][t],gamma[q],beta[q]))
                true.append(test_label[stu][q].item())
        auc.append(metrics.roc_auc_score(true,pred))
    return auc
#Select question and estimation 
def Rank(params,device):
    #data preparation
    triplets = pd.read_csv('data/'+params.data_name+'/train_triples.csv', encoding='utf-8').to_records(index=False)
    metadata = json.load(open('data/'+params.data_name+'/metadata.json', 'r'))
    train_data = AdapTestDataset(triplets,metadata['num_train_students'], metadata['num_questions'])
    test_triplets = pd.read_csv('data/'+params.data_name+'/test_triples.csv', encoding='utf-8').to_records(index=False)
    test_data = AdapTestDataset(test_triplets,metadata['num_test_students'],metadata['num_questions'])
    concept_map = json.load(open('data/'+params.data_name+'/concept_map.json'))
    concept_map={int(k):v for k,v in concept_map.items()}
    
    train_label = np.zeros([metadata['num_train_students'],metadata['num_questions']])-1
    for stu in range(train_data.num_students):
        for k,v in train_data.data[stu].items():
            train_label[stu][k] = v
    test_label = np.zeros([metadata['num_test_students'],metadata['num_questions']])-1
    for stu in range(test_data.num_students):
        for k,v in test_data.data[stu].items():
            test_label[stu][k] = v
    if params.irt_method == 'mcmc':
        beta=np.load('data/'+params.data_name+'/beta.npy')
        gamma=np.load('data/'+params.data_name+'/alpha.npy')
        
        #get collaborative students' abilities
        train_theta_star = []
        for stu in range(train_data.num_students):
            a = []
            b = []
            labels = []
            for q,v in train_data.data[stu].items():
                a.append(gamma[q])
                b.append(beta[q])
                labels.append(train_data.data[stu][q])
            x = scipy.optimize.root(Likelihood, 0,args=(np.array(labels),np.array(a),np.array(b))).x[0]
            if x>4:
                x=4
            if x<-4:
                x=-4
            train_theta_star.append(x)  
        #get tested students' abilities    
        theta_star = []
        for stu in range(test_data.num_students):
            a = []
            b = []
            labels = []
            for q,v in test_data.data[stu].items():
                a.append(gamma[q])
                b.append(beta[q])
                labels.append(test_data.data[stu][q])
            x = scipy.optimize.root(Likelihood, 0,args=(np.array(labels),np.array(a),np.array(b))).x[0]
            if x>4:
                x=4
            if x<-4:
                x=-4
            theta_star.append(x)
        #Complete the records of collaborative students
        for stu in range(train_data.num_students):
            for q in range(train_data.num_questions):
                if q not in train_data.data[stu]:
                    train_label[stu][q] = IRT(train_theta_star[stu],gamma[q],beta[q])
        train_label = torch.FloatTensor(train_label).to(device)
        test_label = torch.FloatTensor(test_label).to(device)
        ga = torch.FloatTensor(gamma).to(device)
    else:
        #get collaborative students' abilities
        irt_model = IRTmodel(train_data.num_questions,train_data.num_students,0.1).to(device)
        irt_model.load_state_dict(torch.load('model/'+params.data_name+'/IRT_GD.pth'))
        train_label = torch.FloatTensor(train_label).to(device)
        test_label = torch.FloatTensor(test_label).to(device)
        op = optim.Adam(irt_model.parameters(), lr=0.1)
        irt_model.alpha.requires_grad = False
        irt_model.beta.requires_grad = False
        irt_model.n_students.data = torch.zeros([train_data.num_students,1]).to(device)
        train_s= (train_label>=0)
        train_y = train_label
        for i in range(10):
            op.zero_grad()
            irt_model.train()
            Pt = irt_model()
            loss = nn.BCELoss()(Pt.flatten()[train_s.flatten()==1],train_y.flatten()[train_s.flatten()==1])
            loss.backward()
            op.step()
            
        #get tested students' abilities  
        train_theta_star = cp.deepcopy(irt_model.n_students.data)
        irt_model = IRTmodel(train_data.num_questions,train_data.num_students,0.1).to(device)
        irt_model.load_state_dict(torch.load('model/'+params.data_name+'/IRT_GD.pth'))
        op = optim.Adam(irt_model.parameters(), lr=0.1)
        irt_model.alpha.requires_grad = False
        irt_model.beta.requires_grad = False
        irt_model.n_students.data = torch.zeros([train_data.num_students,1]).to(device)
        test_s= (test_label>=0)
        test_y = test_label
        test_stu = range(test_data.num_students)
        for i in range(10):
            op.zero_grad()
            irt_model.train()
            Pt = irt_model()[test_stu]
            loss = nn.BCELoss()(Pt.flatten()[test_s.flatten()==1],test_y.flatten()[test_s.flatten()==1])
            loss.backward()
            op.step()
        theta_star = cp.deepcopy(irt_model.n_students.data[test_stu].detach().cpu().numpy().reshape(-1))
        
        #Complete the records of collaborative students
        ga = irt_model.alpha.data.flatten()
        be = irt_model.beta.data.flatten()
        for stu in range(train_data.num_students):
            for q in range(train_data.num_questions):
                if q not in train_data.data[stu]:
                    train_label[stu][q] = P(train_theta_star[stu],ga[q],be[q])
        
        train_theta_star = train_theta_star.detach().cpu().numpy().reshape(-1)
    
    if params.metric_method == 'Ranking':
        if params.irt_method == 'mcmc':
            from selection_strategy import MCMC_Selection as Selection_method
            selection = Selection_method(train_data,test_data,concept_map,train_label,test_label,gamma,beta,params)
        else:
            from selection_strategy import GD_Selection as Selection_method
            selection = Selection_method(train_data,test_data,concept_map,train_label,test_label,irt_model,params)
        selected_questions, stu_theta = selection.get_question()
        iner_rank = Rank_score(train_label,test_label,selected_questions,theta_star,train_theta_star,ga,device)
        
        print('intra Ranking Consistency:',iner_rank)
        rank_result = get_rank_result(train_label,test_label,selected_questions,ga,device).sum(-1)
        inter_rank = pairwise(stu_theta,theta_star)
        inter_rank_c = pairwise(rank_result,theta_star)
        print('inter Ranking Consistency estimated by IRT:',inter_rank)
        print('inter Ranking Consistency estimated by CCAT:',inter_rank_c)  
    if params.metric_method == 'ACC/AUC':
        test_select = []
        test_select_label = cp.deepcopy(test_label)
        np.random.seed(params.seed)
        for stu in range(test_data.num_students):
            select = torch.where(test_label[stu]>=0)[0].cpu().numpy()
            np.random.shuffle(select)
            selects = select[int(0.9*len(select)):]
            test_select_label[stu][selects] = -1
            test_select.append(set(selects))
        if params.irt_method == 'mcmc':
            from selection_strategy import MCMC_Selection as Selection_method
            selection = Selection_method(train_data,test_data,concept_map,train_label,test_select_label,gamma,beta,params)
        else:
            from selection_strategy import GD_Selection as Selection_method
            selection = Selection_method(train_data,test_data,concept_map,train_label,test_select_label,irt_model,params)
        selected_questions, stu_theta = selection.get_question()
        if params.irt_method == 'gd':
            gamma = ga.detach().cpu().numpy()
            beta = be.detach().cpu().numpy()
        ACC = ACC_score(stu_theta,test_select,gamma,beta,test_label)
        print('ACC:',ACC)
        AUC = AUC_score(stu_theta,test_select,gamma,beta,test_label)
        print('AUC:',AUC)
    
if __name__ == '__main__':
    device = torch.device("cuda") if params.device=='cuda' else torch.device("cuda")
    Rank(params,device)
    
    