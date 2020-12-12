#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:27:15 2020

@author: Lantian
"""

import pandas as pd
import numpy as np
from scipy.stats import f
import statsmodels.api as sm
from matplotlib import pyplot as plt

def GRS_test(alpha,eps,mu):
    '''
    T: sample size
    N: number of assets
    L: number of factors
    
    alpha: ndarray, intercepts matrixs, N*1
    eps: ndarray, residual of ols model, T*N
    mu: ndarray, matrix of factor values, T*L
    '''
    T,N = eps.shape
    L = mu.shape[1]
    mu_mean = mu.mean(axis=0)
    COV_e = eps.T.dot(eps)/T
    COV_f = (mu-mu_mean).T.dot((mu-mu_mean))/T
    FGRS = (T-N-L)/N*(alpha.T.dot(np.linalg.inv(COV_e)).dot(alpha)
        )/(1+mu_mean.dot(np.linalg.inv(COV_f)).dot(mu_mean))
    pGRS = 1-f.cdf(FGRS,N,T-N-L)
    return FGRS, pGRS

if __name__=='__main__':
    method = '3-factor'
    if(method=='CAPM'):
        factors = ['Mkt-RF']
        factor_data = pd.read_csv('./F-F_Research_Data_Factors.CSV',index_col = 0)
        factor_data = factor_data[0:1133]
        factor_data.dropna(inplace=True)
        
    elif(method == '3-factor'):
        factors = ['Mkt-RF','SMB','HML']
        factor_data = pd.read_csv('./F-F_Research_Data_Factors.CSV',index_col = 0)
        factor_data = factor_data[0:1133]
        factor_data.dropna(inplace=True)
        
    elif(method == '5-factor'):
        factors = ['Mkt-RF','SMB','HML','RMW','CMA']
        factor_data = pd.read_csv('./F-F_Research_Data_5_Factors_2x3.CSV',index_col = 0)
        factor_data = factor_data[0:690]
        factor_data.dropna(inplace=True)
    
    factor_data = pd.DataFrame(factor_data, dtype='float')
    factor_data.replace(-99.99,0,inplace=True)
    
    return_data = pd.read_csv('./25_Portfolios_5x5.CSV',index_col = 0)
    return_data = return_data[0:1133]
    return_data.dropna(inplace=True)
    return_data = pd.DataFrame(return_data, dtype='float')
    return_data.replace(-99.99,0,inplace=True)
    
    portfolio_list = list(return_data)
    start = ['192607','196307']
    end = ['196306','201612']
    
    for i in range(len(start)):
        alpha = []
        eps = []
        tempfactor = factor_data[start[i]:end[i]]
        tempx = tempfactor[factors]
        time_index = list(tempfactor.index)
        pred = pd.DataFrame(columns=portfolio_list, index=time_index)
        real = pd.DataFrame(columns=portfolio_list, index=time_index)
        for portfolio in portfolio_list:
            tempreturn = return_data[portfolio]
            temp = tempfactor.join(tempreturn)
            tempy = temp[portfolio]-temp['RF']
            real[portfolio] = tempy
            X = sm.add_constant(tempx)
            model = sm.OLS(tempy,X)
            result = model.fit()
            eps.append(list(tempy-result.fittedvalues))
            pred[portfolio] = result.fittedvalues-result.params[0]
            alpha.append(result.params[0])
        eps = pd.DataFrame(eps)
            
        #plot mean eps over portfolios changes with time
        average_eps = eps.mean()
        plt.plot(time_index, average_eps,label='mean eps')
        xtick = [time_index[i] for i in range(0,len(time_index),24)]
        plt.xticks(xtick,rotation=90)
        plt.title('mean of portfolios eps changing with time \n from '+start[i]+' to '+ end[i] +' under '+method)
        plt.legend()
        plt.ylabel('%')
        plt.xlabel('time')
        plt.show()
        
        #plot real average excess return and the predicted
        bar_width = 0.35
        pos = np.arange(len(portfolio_list))
        plt.bar(pos, real.mean(),width = bar_width, label='real excess return')
        plt.bar(pos+bar_width, pred.mean(), width = bar_width, label='predicted excess return',hatch='///')
        plt.xticks(ticks=pos,labels=portfolio_list,rotation=90)
        plt.title('mean real excess returns against the '+ method+ ' predicted \n from '+start[i]+' to'+ end[i])
        plt.legend()
        plt.ylabel('%')
        plt.xlabel('portfolios')
        plt.show()
        
#        plt.bar(pos,real.mean()-pred.mean())
#        plt.xticks(ticks=pos,labels=portfolio_list,rotation=90)
#        plt.title('the difference between real '+ method+ ' and predicted excess return\n from '+start[i]+' to'+ end[i])
#        plt.ylabel('%')
#        plt.xlabel('portfolios')
#        plt.show()
        
        eps1 = np.array(eps).T
        alpha = np.array(alpha)
        fGRS, pGRS = GRS_test(alpha, eps1, np.array(tempx))
        print(fGRS)
        print(pGRS)
        print(start[i]+'-'+end[i])
