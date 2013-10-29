import numpy as np
from matplotlib.pyplot import *

#x = [-7.30000, -4.10000, -1.70000, -0.02564,
#     1.50000, 4.50000, 9.10000]
#y = [-0.80000, -0.50000, -0.20000, 0.00000,
#     0.20000, 0.50000, 0.80000]
#
#coefficients = numpy.polyfit(x, y, 1)
#polynomial = numpy.poly1d(coefficients)
#ys = polynomial(x)
#print coefficients
#print polynomial
#
#plot(x, y, 'o')
#plot(x, ys)
#ylabel('y')
#xlabel('x')
#xlim(-10,10)
#ylim(-1,1)
#show()


from scipy.optimize import curve_fit


def func(t, a, b, c, d):
    x = a*t + b
    y = c*t + d
    return x,y

t = np.linspace(0, 1, 20)

#y = func(t, 1, 2, 3, 4)
#y = np.array(y)
#yn = y + 0.2*np.random.normal(size=len(t))

yn = []
#for i in range(0,len(t):
#    x,y = func(t, 1, 2, 3, 4)
#    yn.append((x+


import pdb; pdb.set_trace()

params, pcov = curve_fit(func, t, yn)


print (params)
