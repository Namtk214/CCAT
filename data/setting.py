# -*- coding: utf-8 -*-
import argparse
import torch

parser = argparse.ArgumentParser(description='KCAT')

parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu')
parser.add_argument('--seed', type=int, default=2024, help='the random seed,2023,2024,2025,2026,2027')
parser.add_argument('--data_name', type=str, default='NIPS2020', help='data name')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--irt_method', type=str, default='gd', help="mcmc,gd")
parser.add_argument('--method', type=str, default='ccat', help="question selection method,random,fsi,kli,maat,ncat,becat,ccat")
params = parser.parse_args()