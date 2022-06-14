# How it works:

The main program is provided with EM.py file, which contains the begin entry for the execution of the package.
It has to be launched on a command shell with the Python3 interpreter, such as "python3 EM.py".

The EM.py script starts to read a dataset from a .csv file given in input and convert it to a Pandas dataframe
to improve the data management and manipulation (a test file has been provided
with the name "RUNEUSDT_4H.csv" which contains all the relevant information for the RUNE/USDT digital asset 
pair obtained from the Binance exchange for a certain period of time ranging from 2020/09/04 to 2021/03/12.
Each report in the dataframe gives information about the digital asset price packed into a 4 hours time window,
such as open time, open price, close price, ecc.

As default parameter, the main program take into acount a 500 close price observations (from the most recent and backward)
and computes the price returns time series that will be modelled by an ARMA-GARCH mixture fitting.
The fitting strategy is obtained by implementing the Expectation Maximization algorithm, which will find the MLE 
for the K gaussian compoenents underlying the observations; the number of iterations to reach a maximum depends on 
the random initial parameters values for the ARMA-GARCH models, so a tollerance parameters needs to be set in order to 
stop the fit when the likelihood doesn't change anymore above that tollerance percentage (toll_param = 0.1%).

At the state of the art the K parameter value can be arbitrary choosen (usually 2 or 3 is a good choice), while the ranks 
of the ARMA and GARCH models (R,S and Q,P respectively) needs to be set to 2 (second order models).
Hopefully future implementations of the package will permit larger values for these orders.
The model orders parameter set (R,S,Q,P,K) is given in input to the GEM1D class contructor, so a change in the parameters
set on the beginning of the main() function will change the model instance of the GEM1D class.

Finally, the "Utils" folder contains all the modules for the arithmetical computation subroutines and functions that 
operates during the fitting procedure, and which are invoked by the main program at each iteration.

At the end of the fit a brief summary statistics will show up and two figures will be visualized one after closing the other:
the first figure contains 3 columns and K+1 rows, showing the variance and the residuals for the whole model along 
the 1st column, the price return modelled series for each of the components in the 2nd column and the modelled variance for
each of the components in the 3rd column; while the 2nd figure shows a zommed in plot of the fitted model upon the real 
analyzed data with the corresponding modelled variance and price returns series. 



