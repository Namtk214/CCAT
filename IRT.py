# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:46:24 2024

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.optim as optim


class IRTmodel(nn.Module):
    def __init__(self, n_question, n_students, lr=0.4):
        """

        Input:
        n_question : int
            questions num.
        n_students : int
            students num.
        lr : TYPE, float
            the learning rate. The default is 0.1.
        """
        super().__init__()
        self.device = torch.device('cuda')
        self.alpha = nn.Parameter(torch.ones([1, n_question]))
        self.beta = nn.Parameter(torch.zeros([1, n_question]))
        self.n_students = nn.Parameter(torch.zeros([n_students, 1]))
        nn.init.xavier_normal_(self.n_students)
        nn.init.xavier_normal_(self.beta)
        nn.init.normal_(self.alpha)
        self.alpha.data = torch.exp(self.alpha)
        self.lr = lr

    def forward(self):
        x = torch.sigmoid(self.alpha*(self.n_students-self.beta))
        return x

    def optim(self, selected_id, y):
        """
        Inouts:
        selected_id : list
            questions id students already test.
        y : tensor
            response of one student.

        """
        self.train()
        op = optim.Adam(self.parameters(), lr=self.lr)
        for i in range(8):
            op.zero_grad()
            x = torch.sigmoid(self.alpha.detach() *
                              (self.n_students[0]-self.beta.detach()))
            x = x.flatten()[selected_id]
            loss = nn.BCELoss()(x, y.flatten()[selected_id])
            loss.backward()
            op.step()

    # return student's ability
    def get_theta(self):
        return self.n_students[0].item()

    # calculate model change in MAAT
    def get_maat(self, selected_id, label):
        self.train()
        op = optim.Adam(self.parameters(), lr=self.lr)
        for i in range(8):
            op.zero_grad()
            x = torch.sigmoid(self.alpha.detach() *
                              (self.n_students[0]-self.beta.detach()))
            x = x.flatten()[selected_id]
            loss = nn.BCELoss()(x, label.flatten()[selected_id])
            loss.backward()
            op.step()

    # calculate the probilites of quesiton that student can correctly answer
    def prob(self, theta):
        return torch.sigmoid(self.alpha*(theta-self.beta))
