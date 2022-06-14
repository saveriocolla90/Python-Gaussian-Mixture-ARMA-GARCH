#--GENERAL USE MODULES--#
import numpy as np

#--STATISTICAL MODULES--#
from scipy.stats import norm


def compute_pZ(sigma, eps, alpha, k, R):
	""" Compute conditional expectation of unobserved data Z """

	#-Numerator from k-th component
	num = (alpha[k]/np.sqrt(sigma[k]))*norm.pdf(eps[k][R:]/np.sqrt(sigma[k]))

	#-Denominator of sum all remaining points
	den = np.zeros(len(sigma[k]))
	for j in range(len(alpha)):
		den += (alpha[j] * norm.pdf( eps[j][R:] / np.sqrt(sigma[j]) )) / np.sqrt(sigma[j])

	#-Return the corresponding fraction
	return (num/den)



def compute_ll(eps, sigma, pZ, alpha, R, N):
	""" Compute the log-likelihood value for the conditional 
	    probability pZ of unobserved variables Z over N observations
	    as expectation value. """

	l1 = np.zeros(len(sigma[0]))
	l2 = np.zeros(len(sigma[0]))
	l3 = np.zeros(len(sigma[0]))

	#-Compute for every component k the 3 parts of the log-likelihood
	for k in range(len(alpha)):
		l1 += np.log(alpha[k])*pZ[k]
		l2 += (np.log(sigma[k])*pZ[k])/2.
		l3 += (pZ[k]*eps[k][R:]**2) / (2.*sigma[k]) 		
	
	#-Return the expectation value 
	return (l1 - l2 - l3).sum()/N


def compute_Q(eps, sigma, pZ, alpha, R):
	""" Compute the log-likelihood value for the conditional 
	    probability pZ of unobserved variables Z over N observations. """

	ll = np.zeros(len(sigma[0]))
	
	#-Sum all over k components
	for k in range(len(alpha)):
		temp_gauss = norm.pdf( eps[k][R:]/np.sqrt(sigma[k]) )
		#-Check if there are zero values and replace them
		if not np.all(temp_gauss):
			temp_gauss = replaceZeros(temp_gauss)
		
		#-Compute Log-likelihood
		ll += pZ[k]*np.log((alpha[k]/np.sqrt(sigma[k]))*temp_gauss)

	return ll.sum()


def replaceZeros(data):
	""" Replace the zero of a array with the lowest 
	    value of the same array. """

	min_nonzero = np.min(data[np.nonzero(data)])
	data[data == 0] = min_nonzero

	return data
