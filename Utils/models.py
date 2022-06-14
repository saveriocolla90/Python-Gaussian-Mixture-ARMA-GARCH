#--GENERAL USE MODULES--#
import numpy as np
import pandas as pd


def ARMA_residuals(y,a,b,R,S):
	""" Compute residuals part from new parameters 
	    for ARMA model """
	
	#-AR part
	y_ARMA = np.zeros(len(y)-R)
	for t in range(R,len(y)):
		y_ARMA[t-R] = np.dot(b, np.array([ y[t-1-i] for i in range(R) ]).T)
		
		#-Check for including MA part
		if t >= (R+S):
			y_ARMA[t-R] += np.dot(a, np.array( [y_ARMA[t-R-1-j]-y[t-1-j] for j in range(S) ]).T)

	#-Return the residuals
	return pd.Series(y_ARMA - y[R:], index=y.index[R:])



def GARCH_sigma(eps,d0,delta,beta,P,Q):
	""" Compute variances series from previous residuals terms eps. """ 
 	
	#-Check for particular low value of residuals (potentialy overflow errors)
	eps[abs(eps) < 1.e-80] = 0.

	#-CH part
	s_GARCH = np.zeros(len(eps)-Q)
	for t in range(Q,len(eps)):
		s_GARCH[t-Q] = d0 + np.dot(delta, np.array([ eps[t-1-i]**2 for i in range(Q) ]).T)	#---overflow encountered in double_scalars

		#-Check for including GAR part
		if t >= (Q+P):
			s_GARCH[t-Q] += np.dot(beta, np.array( [ s_GARCH[t-Q-1-j] for j in range(P) ] ).T)

	#-Return sigma**2 series
	return pd.Series(s_GARCH, index=eps.index[Q:])



def y_arma_model(y,eps,a,b,R,S):
	""" Compute the predicted value "y_hat" modelled with 
	    ARMA(R,S) based on the previous real "y" data and 
	    including information from the residual "eps" of 
	    the model data. """ 

	y_hat = np.zeros(len(eps)-S)
	for t in range(S,len(eps)):
		#-AR part
		y_AR = np.dot(b, np.array([ y[t-i-1] for i in range(R) ]).T)
		
		#-MA part
		eps_MA = np.dot(a, np.array( [eps[t-i-1] for i in range(S) ]).T)

		#-Compute model prediction
		y_hat[t-S] = y_AR + eps_MA
	
	return pd.Series(y_hat, index=eps.index[S:])



def sigma_garch_model(eps,sigma,delta,beta,Q,P):
	""" Compute the predicted value "sigma_hat" modelled with 
	    a GARCH model of order (p,q) based on the previous 
	    residual "eps" data and the "sigma" data. """ 

	sigma_hat = np.zeros(len(sigma)-P)
	for t in range(P,len(sigma)):
		#-AR part
		eps_AR = np.dot(delta,  np.array([ eps[t-i-1]**2 for i in range(Q) ]).T)

		#-CH part
		sigma_CH = np.dot(beta, np.array([ sigma[t-i-1]**2 for i in range(P) ]).T)

		#-Compute model prediction
		sigma_hat[t-P] = eps_AR + sigma_CH 

	return pd.Series(sigma_hat, index=sigma.index[P:])
	

def check_2stability(delta,beta):
	""" Return the parameter value to check the second order stability for
	    the GARCH model with delta and beta parameters. """
	return np.sum(delta)+np.sum(beta) 



def z_roots_module(b,beta):
	""" Solving characteristic equation:

		1 - b_1*z - b_2*z^2 - ... - b_R*z^R = 0

	    in z complex variable for parameters b and beta,
	    related to autoregressive part of ARMA and GARCH model. """ 

	R = len(b)
	P = len(beta)

	if R == 1: mod_z_b = abs(1./b)
	if P == 1: mod_z_beta = abs(1./beta)

	if R == 2:
		z_b = np.zeros(R)
		delta = b[0]**2 + 4*b[1]
		if delta >= 0:
			z_b[0] = (b[0] + np.sqrt(delta)) / (-2.*b[1])   
			z_b[1] = (b[0] - np.sqrt(delta)) / (-2.*b[1])
			
			#-Insert exception: overflow double scalars
			mod_z_b = np.sqrt( z_b[0]**2 + z_b[1]**2 )	#---overflow encountered in double_scalars
		else:
			mod_z_b = np.sqrt( (b[0]/(-2.*b[1]))**2 + (delta/(-2.*b[1]))**2 )

	if P == 2:
		z_beta = np.zeros(P)
		delta = beta[0]**2 + 4*beta[1]
		if delta >= 0:
			z_beta[0] = (beta[0] + np.sqrt(delta)) / (-2.*beta[1])   
			z_beta[1] = (beta[0] - np.sqrt(delta)) / (-2.*beta[1])
			mod_z_beta = np.sqrt( z_beta[0]**2 + z_beta[1]**2 )
		else:
			mod_z_beta = np.sqrt( (beta[0]/(-2.*beta[1]))**2 + (delta/(-2.*beta[1]))**2 )

	return mod_z_b, mod_z_beta
