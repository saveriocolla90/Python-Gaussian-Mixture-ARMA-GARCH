#--GENERAL USE MODULES--#
import numpy as np

#--STATISTICAL MODULES--#
import random

def random_color():
	rgb=[np.random.uniform(0.,1.) for i in range(3)]
	random.shuffle(rgb)
	return tuple(rgb)




