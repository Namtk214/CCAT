# -*- coding: utf-8 -*-
import pandas as pd
import json
import numpy as np
import pystan

from dataset import Dataset, TrainDataset, AdapTestDataset
from setting import *

#Load data
triplets = pd.read_csv(params.data_name+'/train_triples.csv', encoding='utf-8').to_records(index=False)
metadata = json.load(open(params.data_name+'/metadata.json', 'r'))
train_data = AdapTestDataset(triplets,metadata['num_train_students'], metadata['num_questions'])
data_j=[]
data_k=[]
data_y=[]
for j,i in train_data.data.items():
    for k,y in i.items():
        data_j.append(j+1)
        data_k.append(k+1)
        data_y.append(y)

#MCMC init
irt_data = {'J':metadata['num_train_students'],'K':metadata['num_questions'],'N':len(data_y),'jj':data_j,'kk':data_k,'y':data_y}     
irt_code = """
  // saved as irt.stan
  data {
  int<lower=1> J;                     // number of students
  int<lower=1> K;                     // number of questions
  int<lower=1> N;                     // number of observations
  int<lower=0, upper=J> jj[N];  // student for observation n
  int<lower=0, upper=K> kk[N];  // question for observation n
  int<lower=0, upper=1> y[N];   // correctness for observation n
    }
  parameters {
  real mu_beta;                // mean question difficulty
  vector[J] alpha;             // ability for j - mean
  vector[K] beta;              // difficulty for k
  vector<lower=0>[K] gamma;    // discrimination of k
  real<lower=0> sigma_beta;    // scale of difficulties
  real<lower=0> sigma_gamma;   // scale of log discrimination
}
  model {
  alpha ~ std_normal();
  beta ~ normal(0, sigma_beta);
  gamma ~ lognormal(0, sigma_gamma);
  mu_beta ~ cauchy(0, 5);
  sigma_beta ~ cauchy(0, 5);
  sigma_gamma ~ cauchy(0, 5);
  for (n in 1:N) {
  y[n] ~ bernoulli_logit(gamma[kk[n]] * (alpha[jj[n]] - (beta[kk[n]] + mu_beta)));
}
}

"""
irt = pystan.StanModel(model_code = irt_code)
irt_fit=irt.sampling(data = irt_data, iter = 7000, chains = 4,warmup=3000)
beta = irt_fit.extract(permuted=True)['beta']
mu_beta= irt_fit.extract(permuted=True)['mu_beta']
gamma = irt_fit.extract(permuted=True)['gamma']
beta=beta.mean(0)

#save parameter
mu_beta=mu_beta.mean(0)
np.save(params.data_name+'/beta.npy',beta+mu_beta)
gamma=gamma.mean(0)
np.save(params.data_name+'/alpha.npy',gamma)