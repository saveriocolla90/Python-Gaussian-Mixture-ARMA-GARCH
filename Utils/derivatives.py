#--GENERAL USE MODULES--#
import numpy as np
import pandas as pd


def dev_eps(z, a, SR):
	""" Autoregressive derivatives of redisuals. """

	#-Create the empty derivatives list of lists
	dev_eps = [[]]*SR

	#-Orders of back derivatives included 
	S = len(a)

	#-Compute for every parameter order
	for o in range(SR):
		d_eps = np.zeros(len(z)-SR)
		for t in range(SR,len(z)):
			d_eps[t-SR] = -z[t-o-1]
			
			#-Check for including back derivatives part
			if t >= S+SR:
				d_eps[t-SR] -= np.dot(a, np.array( [d_eps[t-SR-1-j] for j in range(S) ]).T)

		#-Join all derivatves into a time series
		dev_eps[o] = pd.Series(d_eps, index=z.index[SR:])
 
	#-Return the complete array of derivatives as pandas series
	return dev_eps




def dev_sigma_ARMA(eps, d_eps, gamma, rho, SR):
	""" Autoregressive derivatives of sigma values. """
	
	#-Create the empty derivatives list of lists
	dev_sigma = [[]]*SR

	#-Defining integration lengths
	Q = len(gamma)
	P = len(rho)

	#-Compute for every parameter order
	for o in range(SR):
		d_sigma = np.zeros(len(d_eps[o])-Q)
		for t in range(Q,len(d_eps[o])):
			#-EPS derivatives part
			d_sigma[t-Q] = 2.*np.dot([eps[t-1-i]*np.exp(gamma[i]) for i in range(Q)], np.array([ d_eps[o][t-1-j] for j in range(Q) ]).T)

			#-Check for including back SIGMA derivatives part
			if t >= Q+P:
				d_sigma[t-Q] += np.dot([np.exp(rho[i]) for i in range(P)], np.array( [d_sigma[t-Q-1-j] for j in range(P) ]).T)

		#-Join all derivatves into a time series
		dev_sigma[o] = pd.Series(d_sigma, index=d_eps[o][Q:].index)
 
	#-Return the complete array of derivatives as pandas series
	return dev_sigma



def dev_sigma_GARCH(z, p1, p2, PQ):
	""" Autoregressive derivatives of sigma values. """
	
	#-Create the empty derivatives list of lists
	dev_sigma = [[]]*PQ

	#-Defining integration lengths
	P = len(p2)

	#-Compute for every parameter order
	for o in range(PQ):
		d_sigma = np.zeros(len(z)-PQ)
		for t in range(PQ,len(z)):
			#-EPS derivatives part
			d_sigma[t-PQ] = np.exp(p1[o])*z[t-o-1]**2

			#-Check for including back SIGMA derivatives part
			if t >= P+PQ:
				d_sigma[t-PQ] += np.dot([np.exp(p2[i]) for i in range(P)], np.array( [d_sigma[t-PQ-1-j] for j in range(P) ]).T)

		#-Join all derivatves into a time series
		dev_sigma[o] = pd.Series(d_sigma, index=z.index[PQ:])
 
	#-Return the complete array of derivatives as pandas series
	return dev_sigma





def y_mix_dev(sigma, d_sigma, de_eps, pZ, O, N):
	""" compute second order derivatives for parameters
	    reletaed to y_hat """

	#-Inlcuding correct start point for EPS arrays
	d_eps = [[]]*O
	for i in range(O):
		d_eps[i] = de_eps[i][O:]
	
	#-Initialize mix derivatives matrix and compute each element
	mix_dev = [np.zeros(O) for o in range(O)]
	for i in range(O):
		for j in range(O):
			mix_dev[i][j] = (-1./N)*( pZ*(0.5*sigma**(-2)*d_sigma[i]*d_sigma[j] + sigma**(-1)*d_eps[i]*d_eps[j]) ).sum()
	
	return np.matrix(mix_dev)

