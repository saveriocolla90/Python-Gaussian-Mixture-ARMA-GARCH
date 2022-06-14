#--GENERAL USE MODULES--#
import numpy as np
import pandas as pd
from collections import deque
import copy
import datetime as dt

#--GRAPHICS MODULES--#
import matplotlib.pyplot as plt

#--STATISTICAL MODULES--#
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

#--LOCAL MODULES--#
from Utils.derivatives import*
from Utils.models import*
from Utils.loglikelihood import*
from Utils.graphic import*



class GEM1D:
	""" Generalized Estimation-Maximization algorithm implementation
	    ------------------------------------------------------------
	    R = AutoRegressive (AR) parameter dimension
	    S = Moveing Average (MA) parameter dimension
	    Q = Generalized AutoRegressive (GAR) parameter dimension
	    P = Conditional Heteroschedasticity (CH) parameter dimension
	    K = Gaussian Mixture number of components
	"""


	def __init__(self, Yt, R, S, P, Q, K):
		""" Initialization of the class by determing the random parameters for
		    ARMA and GARCH model and the corresponding residuals and variance terms. """


		#-Display data range for ARMA model
		print('Analyzed data range:')
		print('\t',Yt.index[0])
		print('\t',Yt.index[-1])
		print("Total: %d data." % len(Yt))
		print("\n-------------------- ROOTS of CHARACTERISTIC EQUATIONs --------------------")

		#-Initialize random parameter values for ARMA-GARCH model in each K component
		ARMA_params = [[]]*K
		GARCH_params = [[]]*K
		print("Initial ARMA parameters:")
		for k in range(K):
			ARMA_params[k] = np.random.uniform(.01,.5,(2,R))	#-[a,b]
			print("k=%d" % k)
			print("\ta:",ARMA_params[k][0])
			print("\tb:",ARMA_params[k][1])

		#-Additive parameter for GARCH model
		delta_0 = np.random.uniform(0.01,.5,K)			
		print("\nInitial GARCH parameters:")
	
		#-A couple of parameters for every k component
		for k in range(K):
			
			#-Generate random parameters values
			GARCH_params[k] = np.random.uniform(.01,.9,(2,Q))	#-[delta,beta] 
				
			#-Check stability condition
			while check_2stability(GARCH_params[k][0],GARCH_params[k][1]) >= 1.:
				GARCH_params[k] = np.random.uniform(.01,.9,(2,Q))	#-[delta,beta] 	
			print("k=%d" % k)
			print("\tDelta_0: %.3f" % delta_0[k])
			print("\tDelta:",GARCH_params[k][0])
			print("\tBeta:",GARCH_params[k][1])
			print("Stability: %.2f" % check_2stability(GARCH_params[k][0],GARCH_params[k][1]))
		
		#-Checking for roots modules of characteristic equations to be outside the unit circle 
		for k in range(K):
			m_z_b, m_z_beta = z_roots_module(ARMA_params[k][1], GARCH_params[k][1])
			print("\nRoots modules for param's component k=%d:" % (k+1))
			print("\t   z_b = %.2f" % m_z_b)
			print("\tz_beta = %.2f" % m_z_beta)

		#-Compute initial ARMA model to get residuals
		eps = [[]]*K
		y_hat = [[]]*K
		for k in range(K):
			eps[k] = ARMA_residuals(Yt, ARMA_params[k][0], ARMA_params[k][1], R, S)
			y_hat[k] = eps[k] + Yt[R:]

		#-Display initial ARMA model
		display = False
		if display == True:
			fig, ax1 = plt.subplots(2,K,figsize=[15,10])
			for k in range(K):
				Yt[R:].plot(ax=ax1[0,k])
				y_hat[k].plot(ax=ax1[0,k])
				eps[k].hist(ax=ax1[1,k],bins=30)
			plt.show()
			quit()

		#-Compute initial GARCH model to get sigma series
		sigma = [[]]*K
		for k in range(K):		
			sigma[k] = GARCH_sigma(eps[k], delta_0[k], GARCH_params[k][0], GARCH_params[k][1], P, Q)

		#-Display initial GARCH model
		display = False
		if display == True:
			fig, ax2 = plt.subplots(2,K,figsize=[15,10])
			for k in range(K):
				ax2[0,k].set_title("Time series - Volatility (k=%d)" % k, fontweight='bold')
				ax2[0,k].axes.get_xaxis().set_visible(False)
				ax2[1,k].set_title("ARMA model residual terms (k=%d)" % k, fontweight='bold')
				ax2[1,k].set_ylabel("Residuals $\epsilon_{t}$")

				Yt[R:].plot(ax=ax2[0,k], label="Returns $r_{t}$",legend=True)
				sigma[k].plot(ax=ax2[0,k], label="Variance $\sigma_{t}^2$", legend=True)
				eps[k].plot(ax=ax2[1,k], color='r', label="Residuals $\epsilon_{t}$", legend=True)

			plt.show()
			quit()

		#-Initialize class variables
		self.Yt = Yt
		self.R = R
		self.S = S
		self.P = P
		self.Q = Q
		self.K = K		
		self.mu = y_hat
		self.eps = eps
		self.sigma = sigma
		self.theta = ARMA_params
		self.omega = GARCH_params
		self.d0 = delta_0
		self.alpha = (1./K)*np.ones(K)
		



	def plot_residuals(self,k,display=True):
		""" Plotting residuals vs. time and their distributions.
		    Return the p-value for a KS-test based on the t-student
		    distribution assumption. """ 

		#----- RESIDUALs visualization -----#
		fig, ax =plt.subplots(1,2,figsize=[15,10])
		_,bins,_ = ax[0].hist(x=self.eps[k], bins=int(len(self.eps[k])/10), density=1)
		dof, loc, scale = stats.t.fit(self.eps[k])
		best_fit_line = stats.t.pdf(bins, dof, loc, scale)
		ax[0].plot(bins, best_fit_line)
		ax[0].set_title("Residual distribution (normalized)", fontweight='bold', fontsize=20)
		ax[0].set_xlabel("\nResidual Price", fontsize=15)
		ax[0].set_ylabel("Density", fontsize=15)
		ax[0].legend(['t-student fit','Data'], fontsize=10)
		D_KS = stats.kstest(self.eps[k], 't', args=[dof, loc, scale], alternative='two-sided')
		print("Testing t-student on RETURNS distribution:")
		print("\t",D_KS)
		self.eps[k][1:].plot(ax=ax[1])
		ax[1].set_title('Residuals vs. Time', fontweight='bold', fontsize=20)
		ax[1].set_xlabel('Close Time (date)', fontsize=15)
		ax[1].set_ylabel('Residual price', fontsize=15)

		if display == True:	plt.show()

		#plot_pacf(x=self.eps**2, alpha=0.05, zero=False)
		#plt.show()

		#-Return p-value of KS test on t-student 
		return D_KS[1]




	def run(self, display_out=True):
		""" Running EM algorithm until convergence to get best parameters values
		    for ARMA and GARCH models. """

		Q_current = []
		Q_var = []
		toll_param = 1.
		limit = False
		while abs(toll_param) > 0.1:

			""" E-step """
			#-First compute pZ
			pZt = [[]]*self.K
			for k in range(self.K):
				pZt[k] = compute_pZ(self.sigma, self.eps, self.alpha, k, self.R)

			#-Defining integration length for log-likelihood derivatives computattion
			N = len(self.Yt) - self.P - self.Q


			""" M-step """
			#-Defining derivatives arrays for both EPS and SIGMA
			d_eps_a = [[]]*self.K
			d_eps_b = [[]]*self.K
			d_sigma_a = [[]]*self.K
			d_sigma_b = [[]]*self.K
			d_sigma_g0 = [[]]*self.K
			d_sigma_g = [[]]*self.K
			d_sigma_r = [[]]*self.K

			#-Defining Hessian matrixes
			M_hess_a = [[]]*self.K
			M_hess_b = [[]]*self.K
			M_hess_g = [[]]*self.K
			M_hess_r = [[]]*self.K

			#-Defining temporaneous incrment arrays
			temp_incr_a = [[]]*self.S
			temp_incr_b = [[]]*self.R
			temp_incr_delta = [[]]*self.Q
			temp_incr_beta = [[]]*self.P
 
			#-Defining moltiplicative factors
			f1 = [[]]*self.K
			f2 = [[]]*self.K
			f1_incr = [[]]*self.K
			f2_incr = [[]]*self.K

			#-Defining temporaneous EPS and SIGMA increment
			eps_incr = [[]]*self.K
			sigma_incr = [[]]*self.K
			d_sigma_g_base = [[]]*self.K
			d_sigma_g0_incr = [[]]*self.K
			d_sigma_g_incr = [[]]*self.K
			d_sigma_r_incr = [[]]*self.K


			#-Iterate over K components
			for k in range(self.K):
				if display_out == True:  print("Computing K=%d  <----START" % k)

				#-Calculate multiplicative factors
				f1[k] = ( (self.eps[k][self.Q:]**2 / self.sigma[k]) -1. ) / (2*self.sigma[k]) 
				f2[k] = self.eps[k][self.Q:] / self.sigma[k]



				#--------------------Updates all THETA and OMEGA parameters----------------------#

				""" Parameter A """
				#-Autoregressive RESIDUALS derivatives over parameter A
				d_eps_a[k] = dev_eps(self.eps[k], self.theta[k][0], self.S)

				#-Autoregressive VARIANCE derivatives over parameter A
				d_sigma_a[k] = dev_sigma_ARMA(self.eps[k][self.S:], d_eps_a[k], np.log(self.omega[k][0]), np.log(self.omega[k][1]), self.S)
								
				#-Compute the 1st log-lokelihood derivative respect A
				d_ll_a = [[]]*self.K			
				for o in range(self.S):
					dll_1 = (pZt[k][self.Q:]*f1[k][self.Q:]*d_sigma_a[k][o]).sum()
					dll_2 = (pZt[k][self.Q:]*f2[k][self.Q:]*d_eps_a[k][o][self.Q:]).sum()
					d_ll_a[k].append( (dll_1 - dll_2)/N )

				#-Compute Hessian matrix for log-likelihood parameter A
				M_hess_a[k] = y_mix_dev(self.sigma[k][self.Q:], d_sigma_a[k], d_eps_a[k], pZt[k][self.Q:], self.S, N)

				#-Check whether the Hessian is invertable
				if np.linalg.det(M_hess_a[k]) != 0:
					with np.errstate(over='raise'):
						try:
							incr_a = np.dot(M_hess_a[k].I, np.array(d_ll_a[k]).T).A1
						except OverflowError  as err:
							if display_out == True:  print("Overflow error encountered in INCR_A: ",err)
							incr_a = np.zeros(self.S)
						except FloatingPointError as err:
							if display_out == True:  print("Overflow error encountered in INCR_DELTA_0: ",err)
							incr_a = np.zeros(self.S)
				else:
					if display_out == True:  print("Det[H_a] = 0 !")
					incr_a = np.zeros(self.S)

				#-If the parameter needs to be incremented
				if np.all(incr_a):
					for o in range(self.S):

						#-Initialize minimum increment
						da = copy.copy(incr_a)

						#-Associate current Q derivative
						dev_Q = d_ll_a[k][o]
						
						#-Creating a mask for increment
						incr_mask = np.zeros(self.S)
						incr_mask[o] = 1.

						#-Update EPS and SIGMA with incremented A value		
						eps_incr[k] = ARMA_residuals(self.Yt, self.theta[k][0]+(da*incr_mask), self.theta[k][1], self.R, self.S)
						sigma_incr[k] = GARCH_sigma(eps_incr[k], self.d0[k], self.omega[k][0], self.omega[k][1], self.P, self.Q)

						#-Update moltiplicative factors
						f1_incr[k] = ( (eps_incr[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 
						f2_incr[k] = eps_incr[k][self.Q:] / sigma_incr[k]

						#-Update derivatives
						d_eps_a[k] = dev_eps(eps_incr[k], self.theta[k][0]+(da*incr_mask), self.S)
						d_sigma_a[k] = dev_sigma_ARMA(eps_incr[k][self.S:], d_eps_a[k], np.log(self.omega[k][0]), np.log(self.omega[k][1]), self.S)

						#-Compute Q derivative
						dev_Q1 = (pZt[k][self.Q:]*(f1_incr[k][self.Q:]*d_sigma_a[k][o] - f2_incr[k][self.Q:]*d_eps_a[k][o][self.Q:])).sum()/N

						#-Iterate untill convergence
						minimum = False
						iteration = 0
						while abs(dev_Q1) < abs(dev_Q) and abs(dev_Q1) > 1.e-5 and iteration <= 10:
							minimum = True
							iteration += 1

							#-Setting new increment
							da += incr_a

							#-Update EPS and SIGMA with incremented A value							
							eps_incr[k] = ARMA_residuals(self.Yt, self.theta[k][0]+(da*incr_mask), self.theta[k][1], self.R, self.S)
							sigma_incr[k] = GARCH_sigma(eps_incr[k], self.d0[k], self.omega[k][0], self.omega[k][1], self.P, self.Q)
		
							#-Update moltiplicative factors
							f1_incr[k] = ( (eps_incr[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 
							f2_incr[k] = eps_incr[k][self.Q:] / sigma_incr[k]

							#-Update derivatives
							d_eps_a[k] = dev_eps(eps_incr[k], self.theta[k][0]+(da*incr_mask), self.S)
							d_sigma_a[k] = dev_sigma_ARMA(eps_incr[k][self.S:], d_eps_a[k], np.log(self.omega[k][0]), np.log(self.omega[k][1]), self.S)

							#-Compute Q derivative
							dev_Q = dev_Q1
							dev_Q1 = (f1_incr[k][self.Q:]*d_sigma_a[k][o] - f2_incr[k][self.Q:]*d_eps_a[k][o][self.Q:]).sum()/N
							if display_out == True:  print(dev_Q1)

						if minimum == True:
							#-Print incremented value of A
							temp_incr_a[o] = (da - incr_a)[o]
							if display_out == True:  
								print("Incrementing A%d:" % (o+1))
								print("\t",self.theta[k][0][o]," -----> ",self.theta[k][0][o]+temp_incr_a[o])
						else:
							temp_incr_a[o] = 0.

					#-Reassign the correct increments array 
					incr_a = temp_incr_a
				if display_out == True:  print("Params A Optimized !\n")
			




				""" Parameter B """
				#-Autoregressive RESIDUALS derivatives over parameter B 
				d_eps_b[k] = dev_eps(self.Yt[self.R:], self.theta[k][0], self.R)

				#-Autoregressive VARIANCE derivatives over parameter B
				d_sigma_b[k] = dev_sigma_ARMA(self.eps[k][self.S:], d_eps_b[k], np.log(self.omega[k][0]), np.log(self.omega[k][1]), self.R)

				#-Compute the 1st log-lokelihood derivative respect A 
				d_ll_b = [[]]*self.K
				for o in range(self.R):
					dll_1 = (pZt[k][self.Q:]*f1[k][self.Q:]*d_sigma_b[k][o]).sum() 
					dll_2 = (pZt[k][self.Q:]*f2[k][self.Q:]*d_eps_b[k][o][self.Q:]).sum()
					d_ll_b[k].append( (dll_1 - dll_2)/N )

				#-Compute Hessian matrix for log-likelihood parameter B
				M_hess_b[k] = y_mix_dev(self.sigma[k][self.Q:], d_sigma_b[k], d_eps_b[k], pZt[k][self.Q:], self.R, N)

				#-Check whether the Hessian is invertable
				if np.linalg.det(M_hess_b[k]) != 0:
					with np.errstate(over='raise'):
						try:
							incr_b = np.dot(M_hess_b[k].I, np.array(d_ll_b[k]).T).A1
						except OverflowError as err:
							if display_out == True:  print("Overflow error encountered in INCR_B: ",err)
							incr_b = np.zeros(self.R)
						except FloatingPointError as err:
							if display_out == True:  print("Overflow error encountered in INCR_DELTA_0: ",err)
							incr_b = np.zeros(self.R)
				else:
					if display_out == True:  print("Det[H_b] = 0 !")
					incr_b = np.zeros(self.R)

			
				#-If the parameter needs to be incremented
				if np.all(incr_b):
					for o in range(self.R):

						#-Initialize minimum increment
						db = copy.copy(incr_b)

						#-Associate current Q derivative
						dev_Q = d_ll_b[k][o]

						#-Creating a mask for increment
						incr_mask = np.zeros(self.R)
						incr_mask[o] = 1.

						#-Check stability condition for ARMA model parameter B
						m_z_b,_ = z_roots_module(self.theta[k][1]+(db*incr_mask), incr_b)
						if m_z_b <= 1.:
							if display_out == True:  print("ARMA parameter B not stable! ---> |z|=%.2f" % m_z_b)
						else:
							#-Update EPS and SIGMA with incremented A value							
							eps_incr[k] = ARMA_residuals(self.Yt, self.theta[k][0], self.theta[k][1]+(db*incr_mask), self.R, self.S)
							sigma_incr[k] = GARCH_sigma(eps_incr[k], self.d0[k], self.omega[k][0], self.omega[k][1], self.P, self.Q)

							#-Update moltiplicative factors
							f1_incr[k] = ( (eps_incr[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 
							f2_incr[k] = eps_incr[k][self.Q:] / sigma_incr[k]

							#-Update derivatives
							d_eps_b[k] = dev_eps(eps_incr[k], self.theta[k][0], self.S)
							d_sigma_b[k] = dev_sigma_ARMA(eps_incr[k][self.S:], d_eps_b[k], np.log(self.omega[k][0]), np.log(self.omega[k][1]), self.S)

							#-Compute Q derivative
							dev_Q1 = (pZt[k][self.Q:]*(f1_incr[k][self.Q:]*d_sigma_b[k][o] - f2_incr[k][self.Q:]*d_eps_b[k][o][self.Q:])).sum()/N

							#-Iterate untill convergence
							minimum = False
							iteration = 0
							while abs(dev_Q1) < abs(dev_Q) and abs(dev_Q1) > 1.e-5 and iteration <= 10:
								minimum = True
								iteration += 1

								#-Setting new increment
								db += incr_b

								#-Check stability condition for ARMA model parameter B
								m_z_b,_ = z_roots_module(self.theta[k][1]+(db*incr_mask), incr_b)
								if m_z_b <= 1.:
									if display_out == True:  print("ARMA parameter B not stable! ---> |z|=%.2f" % m_z_b)
									break

								#-Update EPS and SIGMA with incremented A value							
								eps_incr[k] = ARMA_residuals(self.Yt, self.theta[k][0], self.theta[k][1]+(db*incr_mask), self.R, self.S)
								sigma_incr[k] = GARCH_sigma(eps_incr[k], self.d0[k], self.omega[k][0], self.omega[k][1], self.P, self.Q)
			
								#-Update moltiplicative factors
								f1_incr[k] = ( (eps_incr[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 
								f2_incr[k] = eps_incr[k][self.Q:] / sigma_incr[k]

								#-Update derivatives
								d_eps_b[k] = dev_eps(eps_incr[k], self.theta[k][0], self.S)
								d_sigma_b[k] = dev_sigma_ARMA(eps_incr[k][self.S:], d_eps_b[k], np.log(self.omega[k][0]), np.log(self.omega[k][1]), self.S)

								#-Compute Q derivative
								dev_Q = dev_Q1
								dev_Q1 = (pZt[k][self.Q:]*(f1_incr[k][self.Q:]*d_sigma_b[k][o] - f2_incr[k][self.Q:]*d_eps_b[k][o][self.Q:])).sum()/N
								if display_out == True:  print(dev_Q1)
			
							if minimum == True:
								#-Print incremented value of B
								temp_incr_b[o] = (db - incr_b)[o]
								if display_out == True:  
									print("Incrementing B%d:" % (o+1))
									print("\t",self.theta[k][1][o]," -----> ",self.theta[k][1][o]+temp_incr_b[o])
							else:
								temp_incr_b[o] = 0.

					#-Reassign the correct increments array 
					incr_b = temp_incr_b
				if display_out == True:  print("Params B Optimized !\n")
				
				#-Increment parameters A and B
				self.theta[k][0] += [float(i) for i in incr_a]
				self.theta[k][1] += [float(i) for i in incr_b]
				if display_out == True:  
					print("Current ARMA parameters values:")
					print("\ta: ",self.theta[k][0])
					print("\tb: ",self.theta[k][1])
					print("-------------------------------------------")
				

				
				#-Update EPS and SIGMA with new theta values
				self.eps[k] = ARMA_residuals(self.Yt, self.theta[k][0], self.theta[k][1], self.R, self.S)
				self.sigma[k] = GARCH_sigma(self.eps[k], self.d0[k], self.omega[k][0], self.omega[k][1], self.P, self.Q)

				#-Calculate new multiplicative factor
				f1[k] = ( (self.eps[k][self.Q:]**2 / self.sigma[k]) -1. ) / (2*self.sigma[k]) 


				""" GAMMA updating """
				#-Autoregressive derivatives of variance sigma of GARCH parameters
				d_sigma_g[k] = dev_sigma_GARCH(self.eps[k][self.Q:], np.log(self.omega[k][0]), np.log(self.omega[k][1]), self.Q)

				#-Preparing arrays and series for gamma_0 derivatives
				z0 = pd.Series(np.ones(len(self.eps[k][self.Q:])), index=self.eps[k][self.Q:].index)
				d_sigma_g0[k] = dev_sigma_GARCH(z=z0, p1=[np.log(self.d0[k])], p2=np.log(self.omega[k][1]), PQ=1)
				d_sigma_g0[k][0].drop(d_sigma_g0[k][0].head(1).index, inplace=True)

				#-Appending the gamma_0 derivatives for complete list
				d_sigma_g_base[k] = deque(d_sigma_g[k])
				d_sigma_g_base[k].appendleft(d_sigma_g0[k][0])
				d_sigma_g_base[k] = list(d_sigma_g_base[k])

				#-Compute the 1st log-lokelihood derivative respect omega (inluding updated EPS by theta)
				d_ll_g = [[]]*self.K
				for o in range(self.Q+1):
					d_ll_g[k].append( (pZt[k][self.Q:]*f1[k][self.Q:]*d_sigma_g_base[k][o]).sum() / N )

				#-Compute Hessian matrix for log-likelihood parameters omega (inluding updated theta)
				de_eps0 = [pd.Series(np.zeros(len(self.eps[k][self.Q:])), index=self.eps[k][self.Q:].index) for i in range(self.Q+1)]
				M_hess_g[k] = y_mix_dev(self.sigma[k][self.Q:], d_sigma_g_base[k], de_eps0, pZt[k][self.Q:], self.Q+1, N)




				""" Parameter DELTA_0 """
				#-Check whether the Hessian is invertable
				if np.linalg.det(M_hess_g[k]) != 0:
					with np.errstate(over='raise'):
						try:
							#-Increment
							incr_delta_0 = np.exp(np.dot(M_hess_g[k].I, np.array(d_ll_g[k]).T).A1[0])
						except OverflowError as err:
							if display_out == True:  print("Overflow error encountered in INCR_DELTA_0: ",err)
							incr_delta_0 = 1.	
						except FloatingPointError as err:
							if display_out == True:  print("Overflow error encountered in INCR_DELTA_0: ",err)
							incr_delta_0 = 1.	

				else:
					if display_out == True:  print("Det[H_g] = 0 !")
					incr_delta_0 = 1.

				
				#-If the parameter needs to be incremented
				if incr_delta_0 != 1.:

					#-Initialize minimum increment
					dd0 = copy.copy(incr_delta_0)

					#-Compute Q derivative
					dev_Q = d_ll_g[k][0]

					#-Update EPS and SIGMA with incremented A value							
					sigma_incr[k] = GARCH_sigma(self.eps[k], self.d0[k]/dd0, self.omega[k][0], self.omega[k][1], self.P, self.Q)

					#-Update moltiplicative factors
					f1_incr[k] = ( (self.eps[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 

					#-Compute SIGMA derivatives respect Delta_0 incremented
					d_sigma_g0_incr[k] = dev_sigma_GARCH(z=z0, p1=[np.log(self.d0[k]/dd0)], p2=np.log(self.omega[k][1]), PQ=1)
					d_sigma_g0_incr[k][0].drop(d_sigma_g0_incr[k][0].head(1).index, inplace=True)

					#-Appending the gamma_0 derivatives for complete list
					d_sigma_g_incr[k] = deque(d_sigma_g[k])
					d_sigma_g_incr[k].appendleft(d_sigma_g0_incr[k][0])
					d_sigma_g_incr[k] = list(d_sigma_g_incr[k])

					#-Compute new Q derivative
					dev_Q1 = (pZt[k][self.Q:]*f1_incr[k][self.Q:]*d_sigma_g_incr[k][0]).sum()/N


					#-Iterate untill convergence
					minimum = False
					iteration = 0
					while abs(dev_Q1) < abs(dev_Q) and abs(dev_Q1) > 1.e-5 and iteration <= 10:
						minimum = True
						iteration += 1

						#-Setting new increment
						dd0 += incr_delta_0

						#-Update EPS and SIGMA with incremented A value							
						sigma_incr[k] = GARCH_sigma(self.eps[k], self.d0[k]/dd0, self.omega[k][0], self.omega[k][1], self.P, self.Q)

						#-Update moltiplicative factors
						f1_incr[k] = ( (self.eps[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 

						#-Compute SIGMa derivatives respect Delta_0 incremented
						d_sigma_g0_incr[k] = dev_sigma_GARCH(z=z0, p1=[np.log(self.d0[k]/dd0)], p2=np.log(self.omega[k][1]), PQ=1)
						d_sigma_g0_incr[k][0].drop(d_sigma_g0_incr[k][0].head(1).index, inplace=True)

						#-Appending the gamma_0 derivatives for complete list
						d_sigma_g_incr[k] = deque(d_sigma_g[k])
						d_sigma_g_incr[k].appendleft(d_sigma_g0_incr[k][0])
						d_sigma_g_incr[k] = list(d_sigma_g_incr[k])

						#-Compute new Q derivative
						dev_Q = dev_Q1
						dev_Q1 = (pZt[k][self.Q:]*f1_incr[k][self.Q:]*d_sigma_g_incr[k][0]).sum()/N
						if display_out == True:  print(dev_Q1)

					if minimum == True:
						#-Print incremented value of DELTA_0
						incr_delta_0 = dd0 - incr_delta_0
						if display_out == True:  
							print("Incrementing Delta_0:")
							print("\t",self.d0[k]," -----> ",self.d0[k]/incr_delta_0)
					else:
						incr_delta_0 = 1.
				if display_out == True:  print("Param Delta_0 Optimized !\n")
			




				""" Parameter DELTA """
				#-Check whether the Hessian is invertable
				if np.linalg.det(M_hess_g[k]) != 0:
					with np.errstate(over='raise',divide='raise'):
						try:
							#-Increment 
							incr_delta = np.exp(np.dot(M_hess_g[k].I, np.array(d_ll_g[k]).T).A1[1:])

							#-Check if increment leads to feaseble values
							while any(item >= 1. for item in (self.omega[k][0]/incr_delta)) or np.sum((self.omega[k][0]/incr_delta)) >=1. :
								incr_delta *= 2.
								if display_out == True:  print("---> Found instability for DELTA! Needs to incrment M.F.:", incr_delta)

						except OverflowError as err:
							if display_out == True:  print("Overflow error encountered in INCR_DELTA: ",err)
							incr_delta = np.ones(self.Q)	
						except FloatingPointError as err:
							if display_out == True:  print("Overflow error encountered in INCR_DELTA: ",err)
							incr_delta = np.ones(self.Q)	
						except ZeroDivisionError as err:
							if display_out == True:  print("Overflow error encountered in INCR_DELTA: ",err)
							incr_delta = np.ones(self.Q)			
				else:
					if display_out == True:  print("Det[H_g] = 0 !")
					incr_delta = np.ones(self.Q)
				
				#-If the parameter needs to be incremented
				if not all(it == 1. for it in incr_delta):
					for o in range(self.Q):
						#-Initialize minimum increment
						dd = copy.copy(incr_delta)

						#-Associate current Q derivative
						dev_Q = d_ll_g[k][o]
						
						#-Creating a mask for increment
						incr_mask = np.ones(self.Q)/incr_delta
						incr_mask[o] = 1.

						#-Update EPS and SIGMA with incremented A value							
						sigma_incr[k] = GARCH_sigma(self.eps[k], self.d0[k], self.omega[k][0]/(dd*incr_mask), self.omega[k][1], self.P, self.Q)

						#-Update moltiplicative factors
						f1_incr[k] = ( (self.eps[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 

						#-Autoregressive derivatives of variance sigma of GARCH parameters
						d_sigma_g_incr[k] = dev_sigma_GARCH(self.eps[k][self.Q:], np.log(self.omega[k][0]/(dd*incr_mask)), np.log(self.omega[k][1]), self.Q)

						#-Compute new Q derivative
						dev_Q1 = (pZt[k][self.Q:]*f1_incr[k][self.Q:]*d_sigma_g_incr[k][o]).sum()/N
						
						#-Iterate untill convergence
						minimum = False
						iteration = 0
						while abs(dev_Q1) < abs(dev_Q) and abs(dev_Q1) > 1.e-5 and iteration <= 10:
							minimum = True
							iteration += 1 

							#-Setting new increment
							dd += incr_delta

							#-Resetting the mask
							incr_mask = np.ones(self.P)/dd
							incr_mask[o] = 1.

							#-Check if increment leads to feaseble values							
							if any(item >= 1. for item in (self.omega[k][0]/dd)) or np.sum((self.omega[k][0]/dd)) >=1. :
								if display_out == True:  print("Brake DELTA cycle")
								break

							#-Update EPS and SIGMA with incremented A value							
							sigma_incr[k] = GARCH_sigma(self.eps[k], self.d0[k], self.omega[k][0]/(dd*incr_mask), self.omega[k][1], self.P, self.Q)

							#-Update moltiplicative factors
							f1_incr[k] = ( (self.eps[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 

							#-Autoregressive derivatives of variance sigma of GARCH parameters
							d_sigma_g_incr[k] = dev_sigma_GARCH(self.eps[k][self.Q:], np.log(self.omega[k][0]/(dd*incr_mask)), np.log(self.omega[k][1]), self.Q)

							#-Compute new Q derivative
							dev_Q = dev_Q1
							dev_Q1 = (pZt[k][self.Q:]*f1_incr[k][self.Q:]*d_sigma_g_incr[k][o]).sum()/N

						if minimum == True:
							#-Get back the previous increment
							temp_incr_delta[o] = (dd - incr_delta)[o]

							#-Print incremented value of DELTA
							if display_out == True:  
								print("Incrementing Delta_%d:" % (o+1))
								print("\t",self.omega[k][0][o]," -----> ",self.omega[k][0][o]/temp_incr_delta[o])
						else:
							temp_incr_delta[o] = 1.

					#-Reassign the correct increments array 
					incr_delta = temp_incr_delta
				if display_out == True:  print("Params Delta Optimized !\n")





				""" RHO updating """
				#-Autoregressive derivatives of variance sigma of GARCH parameters
				d_sigma_r[k] = dev_sigma_GARCH(self.sigma[k], np.log(self.omega[k][1]), np.log(self.omega[k][1]), self.P)

				#-Compute the 1st log-lokelihood derivative respect omega (inluding updated EPS by theta)
				d_ll_r = [[]]*self.K
				for o in range(self.P):
					d_ll_r[k].append( (pZt[k][self.Q:]*f1[k][self.Q:]*d_sigma_r[k][o]).sum() / N )

				#-Compute Hessian matrix for log-likelihood parameters omega (inluding updated theta)
				de_eps0 = [pd.Series(np.zeros(len(self.eps[k][self.Q:])), index=self.eps[k][self.Q:].index) for i in range(self.P)]
				M_hess_r[k] = y_mix_dev(self.sigma[k][self.Q:], d_sigma_r[k], de_eps0, pZt[k][self.Q:], self.P, N)




				""" Parameter BETA """
				#-Check whether the Hessian is invertable
				if np.linalg.det(M_hess_r[k]) != 0:
					with np.errstate(over='raise',divide='raise'):
						try:
							#-Increment
							incr_beta = np.exp(np.dot(M_hess_r[k].I, np.array(d_ll_r[k]).T).A1)
				
							#-Check if increment leads to feaseble values
							while any(item >= 1. for item in (self.omega[k][1]/incr_beta)) or np.sum((self.omega[k][1]/incr_beta)) >=1. :
								incr_beta *= 2.
								if display_out == True:  print("---> Found instability for BETA! Needs to incrment M.F.:", incr_beta)

						except OverflowError as err:
							if display_out == True:  print("Overflow error encountered in INCR_BETA: ",err)
							incr_beta = np.ones(self.P)				
						except FloatingPointError as err:
							if display_out == True:  print("Overflow error encountered in INCR_BETA: ",err)
							incr_beta = np.ones(self.P)	
						except ZeroDivisionError as err:
							if display_out == True:  print("Overflow error encountered in INCR_BETA: ",err)
							incr_beta = np.ones(self.P)				
				else:
					if display_out == True:  print("Det[H_r] = 0 !")
					incr_beta = np.ones(self.P)
				
				#-If the parameter needs to be incremented
				if not all(it == 1. for it in incr_beta):
					for o in range(self.P):
						#-Initialize minimum increment
						dbe = copy.copy(incr_beta)

						#-Associate current Q derivative
						dev_Q = d_ll_r[k][o]
					
						#-Creating a mask for increment
						incr_mask = np.ones(self.P)/incr_beta
						incr_mask[o] = 1.

						#-Check for stability for GARCH the model
						_,m_z_beta = z_roots_module(incr_delta, self.omega[k][1]/(dbe*incr_mask))
						if m_z_beta <= 1. :
							if display_out == True:  print("GARCH parameter Beta not stable! ---> |z|=%.2f" % m_z_beta)

						stab = check_2stability(self.omega[k][0]/incr_delta,self.omega[k][1]/(dbe*incr_mask))
						if stab >= 1. :
							if display_out == True:  
								print("Exploding increment for Beta_%d !" % (o+1))
								print("Instability: %.2f" % stab)

						#-Trying to get back the stability condition
						n=0
						while m_z_beta <= 1. or stab >= 1. : 
							n+=1
							if display_out == True:  
								print((self.omega[k][1]/(dbe*incr_mask))[o])
								print(dbe[o])
							incr_mask[o] = 2.**n
							if display_out == True:  
								print(incr_mask[o])
								print("\tReduce to: %.4f" % ((self.omega[k][1]/(dbe*incr_mask))[o]))
							stab = check_2stability(self.omega[k][0]/incr_delta,self.omega[k][1]/(dbe*incr_mask))
							if display_out == True:  print("\tStability:",stab)
							if n == 10:
								limit = True
								if display_out == True:  print("Reached limit! Continue without increment.")
								break
						if limit == True:
							limit = False
							temp_incr_beta[o] = 1.
							continue

						#-Update EPS and SIGMA with incremented A value							
						sigma_incr[k] = GARCH_sigma(self.eps[k], self.d0[k], self.omega[k][0], self.omega[k][1]/(dbe*incr_mask), self.P, self.Q)

						#-Update moltiplicative factors
						f1_incr[k] = ( (self.eps[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 

						#-Autoregressive derivatives of variance sigma of GARCH parameters
						d_sigma_r_incr[k] = dev_sigma_GARCH(self.sigma[k], np.log(self.omega[k][1]), np.log(self.omega[k][1]/(dbe*incr_mask)), self.P)

						#-Compute new Q derivative
						dev_Q1 = (pZt[k][self.Q:]*f1_incr[k][self.Q:]*d_sigma_r_incr[k][o]).sum()/N

						#-Iterate untill convergence
						iteration = 0
						minimum = False
						while abs(dev_Q1) < abs(dev_Q) and abs(dev_Q1) > 1.e-5 and iteration <= 10:
							minimum = True
							iteration += 1

							#-Setting new increment
							dbe += incr_beta
							
							#-Resetting the mask
							incr_mask = np.ones(self.P)/dbe
							incr_mask[o] = 2.**n

							#-Check for stability for GARCH the model
							_,m_z_beta = z_roots_module(incr_delta, self.omega[k][1]/(dbe*incr_mask))
							if m_z_beta <= 1.:
								if display_out == True:  print("GARCH parameter Beta not stable! ---> |z|=%.2f" % m_z_beta)
								n=0				

							#-Check for stability for GARCH the model
							stab = check_2stability(self.omega[k][0]/incr_delta,self.omega[k][1]/(dbe*incr_mask))
							if stab >= 1. :
								if display_out == True:  
									print("Exploding increment for Beta_%d !" % (o+1))
									print("Instability: %.2f" % stab)
								n=0				

							#-Trying to get back the stability condition
							while m_z_beta <= 1. or stab >= 1. : 
								n+=1
								if display_out == True:  
									print((self.omega[k][1]/(dbe*incr_mask))[o])
									print(dbe[o])
								incr_mask[o] = 2.**n
								if display_out == True:  
									print(incr_mask[o])
									print("\tReduce to: %.4f" % ((self.omega[k][1]/(dbe*incr_mask))[o]))
								stab = check_2stability(self.omega[k][0]/incr_delta,self.omega[k][1]/(dbe*incr_mask))
								if display_out == True:  print("\tStability:",stab)
								if n == 10:
									if display_out == True:  print("Reached limit! Continue without increment.")
									limit = True
									break
							if limit == True:
								break

							#-Update EPS and SIGMA with incremented A value							
							sigma_incr[k] = GARCH_sigma(self.eps[k], self.d0[k], self.omega[k][0], self.omega[k][1]/(dbe*incr_mask), self.P, self.Q)

							#-Update moltiplicative factors
							f1_incr[k] = ( (self.eps[k][self.Q:]**2 / sigma_incr[k]) -1. ) / (2*sigma_incr[k]) 

							#-Autoregressive derivatives of variance sigma of GARCH parameters
							d_sigma_r_incr[k] = dev_sigma_GARCH(self.sigma[k], np.log(self.omega[k][1]), np.log(self.omega[k][1]/(dbe*incr_mask)), self.P)
							
							#-Compute new Q derivative
							dev_Q = dev_Q1
							dev_Q1 = (pZt[k][self.Q:]*f1_incr[k][self.Q:]*d_sigma_r_incr[k][o]).sum()/N
							if display_out == True:  print(dev_Q1)

						if limit == True:
							limit = False
							temp_incr_beta[o] = 1.
							continue

						if minimum == True:
							#-Get back the previous increment
							temp_incr_beta[o] = (dbe - incr_beta)[o]*incr_mask[o]						

							#-Print incremented value of BETA
							if display_out == True:  
								print("Incrementing Beta_%d:" % (o+1))
								print("\t",self.omega[k][1][o]," -----> ",self.omega[k][1][o]/temp_incr_beta[o])
						else:
							temp_incr_beta[o] = 1.

					#-Reassign the correct increments array 
					incr_beta = temp_incr_beta
				if display_out == True:  print("Params Beta Optimized !\n")

				#-Increment parameters DELTA_0, DELTA and BETA
				self.d0[k] /= float(incr_delta_0) 
				self.omega[k][0] /= [float(i) for i in incr_delta]
				self.omega[k][1] /= [float(i) for i in incr_beta]
				if display_out == True:  
					print("Current GARCH parameters values:")
					print("\tDelta_0: %.4f" % self.d0[k])
					print("\tDelta: ",self.omega[k][0])
					print("\tBeta : ",self.omega[k][1])
					print("-------------------------------------------")

				#-Compute Q for comparison	
				Q1 = compute_Q(self.eps, self.sigma, pZt, self.alpha, self.R)
				if display_out == True:  print(Q1)

				#-Update SIGMA with new omega values
				self.sigma[k] = GARCH_sigma(self.eps[k], self.d0[k], self.omega[k][0], self.omega[k][1], self.P, self.Q)						

				#-Evaluate Q log-likelihood function on current parameters value with a gaussian distribution
				Q2 = compute_Q(self.eps, self.sigma, pZt, self.alpha, self.R)
				if display_out == True:  
					print(Q2)
					print("Computation for K=%d  <----END" % k)

			#-Update alphas
			for k in range(self.K):

				#-Expectation values
				self.alpha[k] = np.sum(pZt[k])/N
				if display_out == True:  print("Alpha k=%d: %.2f" % (k,self.alpha[k]))
						
			if display_out == True:  print("Done!\n")

				
			#-Evaluate Q log-likelihood function on current parameters values 
			Q_current.append(compute_Q(self.eps, self.sigma, pZt, self.alpha, self.R))
		
			#-Update tollerance parameter
			if len(Q_current) > 1 :
				toll_param = 100.*(Q_current[-1] - Q_current[-2])/Q_current[-2]

			#-Display current Q value
			if display_out == True:  
				print("------------------------------------------------------------------------")
				print('\t\t     Current Q value: %.3f' % Q_current[-1])
				print('\t\t           Variation: %.3f %%' % toll_param)
				print("------------------------------------------------------------------------\n")


		#-Create complete parameters dataframe for final results
		df_param = [[]]*self.K 
		for k in range(self.K):
			#-Initilize DF
			dfp = pd.DataFrame.from_dict({'a':[0],'b':[0], 'delta':self.d0[k], 'beta':[0]})

			#-Preparing column arrays
			a = np.array(self.theta[k][0]).T
			b = np.array(self.theta[k][1]).T
			delta = np.array(self.omega[k][0]).T
			beta = np.array(self.omega[k][1]).T

			#-Append data arrays to DF
			for i in range(self.R):
				dfp = dfp.append({'a':a[i],'b':b[i], 'delta':delta[i], 'beta':beta[i]},ignore_index=True)
			df_param[k] = dfp.assign(Alpha=self.alpha[k]).set_index('Alpha',append=True).swaplevel(0,1)
		
		#-Join all DFs in one
		df_param_tot = df_param[0]
		for k in range(1,self.K):
			df_param_tot = df_param_tot.append(df_param[k])

		#-Return the Parameters DF
		return df_param_tot, self.eps




	def plot_result(self, price, y_model, y_hat, std_model, symbol=None, display=False, display_big=False):
		""" Plot the model results """ 
		
		#-Adding INNOVATION terms
		innovation = self.Yt[self.R+self.S:] - y_model
		#print(innovation.var())
		eps_w = std_model*innovation
		#y += eps_w


		#-MSE
		mse = mean_squared_error(self.Yt[self.R+self.S:], y_model)
		print("Model MSE: %.4f" % mse)

		MSE = 0.
		for i in range(len(eps_w)):
			MSE += eps_w[i]**2.
		print("Residuals MSE: %.3f" % np.sqrt(MSE/len(eps_w)))
		
		#-Converting to REAL price values
		shift = self.R+self.S
		price_hat = np.zeros((len(price)-shift)+1)
		price_hat[0] = price[shift-1] 	
		for t in range(shift+1,len(price)+1):
			price_hat[t-shift] = price[t-1]*((abs(y_model[t-shift-1])*np.sign(y_model[t-shift-1] - y_model[t-shift-2])/100.)+1.)  
		price_hat = pd.Series(price_hat[1:], index=price.index[shift:])

		#-Price MSE
		price_MSE = 0.
		for i in range(len(price_hat)):
			price_MSE += (price[i+shift] - price_hat[i])**2.
		print("Model Price MSE: %.4f" % (1000.*np.sqrt(price_MSE/len(price_hat))))


		if display == True:		
			#-Generating grid plots
			fig = plt.figure(constrained_layout=False,figsize=[15,10])
			gs = fig.add_gridspec(ncols=3,nrows=self.K+1,hspace=.0, wspace=.25)
			ax1 = fig.add_subplot(gs[0,0])
			ax2 = fig.add_subplot(gs[1,0])
			ax3 = fig.add_subplot(gs[2,0])

			ax_returns = [[]]*(self.K+1)
			ax_variance = [[]]*(self.K+1)
			for k in range(self.K+1):
				ax_returns[k] = fig.add_subplot(gs[k,1])
				ax_variance[k] = fig.add_subplot(gs[k,2])

			""" 1st Column Plots """
			#-Close Price 
			price[shift:].plot(ax=ax1, label="Real Data $y_{t}$", legend=True, fontsize=15)
			price_hat.plot(ax=ax1, label="Model Data $\hat{y}_{t}$", legend=True)

			#-Residuals
			eps_w.plot(ax=ax2, color='k', label="Residuals $\epsilon_{t}$", legend=True)

			#-Standard deviation
			std_model.plot(ax=ax3, color='r', label="Weigthed STD  $\sigma_{t}$", legend=True)


			""" 2nd & 3rd Columns Plots """
			self.Yt[shift:].plot(ax=ax_returns[0], label="Real Data $r_{t}$", legend=True)
			y_model.plot(ax=ax_returns[0], label="Model Data $\hat{r}_{t}$", legend=True)
			(y_model + std_model).plot(ax=ax_returns[0], color='g', label="$\hat{r}_{t} + \sigma_{t}$")
			(y_model - std_model).plot(ax=ax_returns[0], color='g', label="$\hat{r}_{t} + \sigma_{t}$")
			std_model.plot(ax=ax_returns[0], color='k')
			(0.-std_model).plot(ax=ax_returns[0], color='k')
			
			(std_model**2.).plot(ax=ax_variance[0], label="Estimated variance ${\hat{\sigma}_{t}}^2$", legend=True)
			
			for k in range(self.K):
				y_hat[k].plot(ax=ax_returns[k+1], color=random_color(), label="k=%d" % (k+1), legend=True)
				ax_returns[k+1].set_ylabel("Component k=%d\n$\\alpha_{%d}=%.2f$" % (k+1,k+1,self.alpha[k]), fontsize=10)

				self.sigma[k].plot(ax=ax_variance[k+1], color=random_color(), label="k=%d" % (k+1), legend=True)
				ax_variance[k+1].set_ylabel("Component k=%d\n$\\alpha_{%d}=%.2f$" % (k+1,k+1,self.alpha[k]), fontsize=10)			

			#-Titles and labels
			fig.suptitle("ARMA(%d,%d)-GARCH(%d,%d) model K=%d" % (self.R,self.S,self.Q,self.P,self.K), fontweight="bold", fontsize=20)
			ax1.set_title("CLOSE PRICE", fontweight="bold", fontsize=15)
			ax1.set_ylabel("Close Price (USDT)\n",fontsize=15)
			ax2.set_ylabel("Residuals  $\epsilon_{t}$",fontsize=15)
			ax3.set_ylabel("Weigthed STD  $\sigma_{t}$",fontsize=15)
			ax_returns[0].set_title("RETURNS", fontweight="bold", fontsize=15)
			ax_returns[0].set_ylabel("Returns  $r_{t}$",fontsize=15)
			ax_variance[0].set_title("VARIANCE", fontweight="bold", fontsize=15)
			ax_variance[0].set_ylabel("Model variance  ${\hat{\sigma}_{t}}^2$",fontsize=15)

			#-Plot horrizzontals lines in RETURNS plot
			for k in range(self.K+1):
				ax_returns[k].axhline(linestyle='--', color='k')
			
			plt.show()
		
		if display_big == True:
			#-Generating grid plots
			fig2, ax = plt.subplots(1,1,figsize=[15,10])

			#-Plot return model over real data and variance prediction			
			self.Yt[shift:].plot(ax=ax, label="Real Data $r_{t}$", legend=True)
			y_model.plot(ax=ax, label="Model Data $\hat{r}_{t}$", legend=True)
			(y_model + std_model).plot(ax=ax, color='g', label="$\hat{r}_{t} \pm \sigma_{t}$", legend=True)
			(y_model - std_model).plot(ax=ax, color='g', legend=False)
			std_model.plot(ax=ax, color='k', label="$\pm \sigma_{t}$ model", legend=True)
			(0.-std_model).plot(ax=ax, color='k')

			#-Titles and labels
			fig2.suptitle("ARMA(%d,%d)-GARCH(%d,%d) model K=%d" % (self.R,self.S,self.Q,self.P,self.K), fontweight="bold", fontsize=20)
			if symbol != None:
				ax.set_title(str(symbol+" Returns"), fontweight="bold", fontsize=15)
			else:
				ax.set_title("RETURNS", fontweight="bold", fontsize=15)

			ax.set_ylabel("Returns  $r_{t}$",fontsize=15)
			ax.axhline(linestyle='--', color='k')
			
			plt.show()



	def predict(self, horizon, close, t_delta, new_y=False, new_sigma=False):
		""" Forcast data with input horizon window """

		#-Initialize forcast date arrays
		date = self.Yt.index[-1]
		y_real = self.Yt[-self.R-1:]
		eps_real = [[]]*self.K
		sigma_real = [[]]*self.K
	
		
		#-Generate EPS and SIGMA array data 
		for k in range(self.K):
			eps_real[k] = self.eps[k][-self.S-1:]
			sigma_real[k] = self.sigma[k][-self.P-1:]

		#-Adding horizion date positions
		for i in range(horizon):
			date += t_delta 
			y_real[date] = 1.
			for k in range(self.K):
				eps_real[k][date] = 1.
				sigma_real[k][date] = 1.
		
		#-Build up ARMA model prediction
		y_hat = [[]]*self.K
		for k in range(self.K):
			y_hat[k] = y_arma_model(y_real, eps_real[k], self.theta[k][0], self.theta[k][1], self.R, self.S)

		#-Creating resulting RETURNs series
		prediction = self.alpha[0]*y_hat[0]
		for k in range(1,self.K):
			prediction += self.alpha[k]*y_hat[k]

		if new_y == True:
			return prediction
			
		#-Build up GARCH model
		sigma_hat = [[]]*self.K
		for k in range(self.K):
			sigma_hat[k] = sigma_garch_model(eps_real[k], sigma_real[k], self.omega[k][0], self.omega[k][1], self.Q, self.P)

		#-Weighted variance
		var1 = self.alpha[0]*sigma_hat[0]
		for k in range(1,self.K):
			var1 += self.alpha[k]*sigma_hat[k]

		var2 = self.alpha[0]*(y_hat[0] - prediction)**2
		for k in range(1,self.K):
			var2 += self.alpha[k]*(y_hat[k] - prediction)**2

		#-Adding up both variance weights
		std = np.sqrt(var1 + var2)
		if new_sigma == True:
			return std
		print("Return prediction: ",prediction[-1]," +/- ",std[-1])

		#-Split last two excpected values for better visualization		
		excpected = copy.copy(y_real[-2:])
		excpected[-1] = close
		
		#-Real data - model data offset
		offset = excpected[-2] - prediction[-2]

		if np.sign(excpected[-1] - excpected[-2]) == np.sign(prediction[-1] - prediction[-2]):
			print("CORRECT !!")
		else:
			print("Incorrect... :(")

		#-Plot Prediction
		fig, ax = plt.subplots(1,1,figsize=[8,5])
		y_real[:-horizon].plot(ax=ax, label="Real Returns", legend=True)
		excpected.plot(ax=ax, color="g", marker="*", label="Excpected", legend=True)
		ax.errorbar(y_real.index[-1], prediction[-1], yerr=std[-1], marker='s', label="Prediction $r_{t}$", color='b')
		ax.errorbar(y_real.index[-2], prediction[-2], yerr=std[-2], marker='s', label="Prediction $r_{t-1}$")
		ax.axhline(linestyle='--',color='k')
		plt.legend(loc="upper left")
		plt.show()



	def compute_variation(self):
		""" Compute the current variance estimation """
		#-Build up ARMA model
		y_hat = [[]]*self.K
		for k in range(self.K):
			y_hat[k] = y_arma_model(self.Yt[self.R:], self.eps[k], self.theta[k][0], self.theta[k][1], self.R, self.S)

		#-Creating resulting RETURNs series
		y_model = self.alpha[0]*y_hat[0]
		for k in range(1,self.K):
			y_model += self.alpha[k]*y_hat[k]

		#-Weighted variance
		var1 = self.alpha[0]*self.sigma[0]
		for k in range(1,self.K):
			var1 += self.alpha[k]*self.sigma[k]

		var2 = self.alpha[0]*(y_hat[0] - y_model)**2
		for k in range(1,self.K):
			var2 += self.alpha[k]*(y_hat[k] - y_model)**2

		#-Adding up both variance weights
		std = np.sqrt(var1 + var2)
		
		return y_model, y_hat, std

		

	def summary(self):
		""" Visualize summary statistics of the convergence cycle
		    and fitted parameters values. """

		""" WORKING PROGRES... """

		




def main():
	
	#-Path to CSV file
	csv_path = './RUNEUSDT_4H.csv'
		
	#-Aux costants
	n_data_fit = 500
	
	#-MODEL parameters dimensions
	R = 2
	S = 2
	P = 2
	Q = 2
	K = 2
	
	#-Initialize the start time
	startTime = dt.datetime.now()
	print("Execution Start time (UTC+1):",startTime.strftime("%Y-%m-%d %H:%M:%S"))
	print("Reading CSV file to analyze:",csv_path)

	#-Reading CSV file to analize
	CSV_read = pd.read_csv(csv_path).set_index('Open Time')
	CSV_read.index = pd.to_datetime(CSV_read.index/1000, unit='s')
	print("Done!\n")

	#-Date ranges and RETURNS series
	df_analysis = CSV_read.iloc[-n_data_fit:]
	close_returns = 100.*df_analysis.Close.pct_change().dropna()
	print("Analyzed data range (UTC):")
	print(close_returns.index[0].strftime("%Y-%m-%d %H:%M")," ---> ",close_returns.index[-1].strftime("%Y-%m-%d %H:%M"))
	print("Total data: ", len(close_returns))

	#-Initialize the EM class with model orders (R,S,Q,P) and components K
	EM = GEM1D(close_returns, R=R, S=S, P=P, Q=Q, K=K)	

	print("==========================================================================")
	print("------------------------ START MLE OPTIMIZZATION -------------------------")
	print("==========================================================================")
	print("\n>   COMPUTING . . .")

	#-Running the MLE optimizzation algorithm
	start_time_model = dt.datetime.now() 
	df_model_param, residuals = EM.run(display_out=False)
	end_time_model = dt.datetime.now()

	#-Print summary of results
	print("\nPARAMETERS RESULT:")
	print(df_model_param)
	print("==========================================================================")
	print("Complete Execution at time:", end_time_model.strftime("%Y-%m-%d %H:%M:%S"))
	print("Total Execution Time:", (end_time_model - start_time_model))

	#-Return the fitted model +/- standard deviation (STD)
	y_model, y_hat, std_model = EM.compute_variation()
	y_std_max = y_model[-1]+std_model[-1]
	y_std_min = y_model[-1]-std_model[-1]
	print("Last Return model: %.5f +/- %.5f  [%.5f , %.5f]" % (y_model[-1], std_model[-1], y_std_min, y_std_max) )

	#-Plot model results	
	EM.plot_result(df_analysis.Close[1:-1], y_model, y_hat, std_model, display=True, display_big=True)
	
	

if __name__=="__main__":
	main()















