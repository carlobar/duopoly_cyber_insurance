import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib import cm
import pdb
import tools_gt as gt

def savefig(title):
    dir_images = './images/'
    try:
        plt.savefig(dir_images + title, bbox_inches = 'tight')
    except:
        plt.savefig(title, bbox_inches = 'tight')


def investment(m):
    return np.log(m0/m) / np.log(alpha)

def J(m1, m2):
    x1 = sigma + m1
    x2 = sigma + m2
    return a**2/(9*b) + sigma**2/b * (2*x2-sigma)**2 * x1 / (4*x1*x2-sigma**2)**2 - investment(m1)

def J_1(m1, m2):
    return J(m1, m2)

def J_2(m1, m2):
    return J(m2, m1)

def sigma_tilde(m0):
    gamma = m0*np.log(alpha) / b * (5/3)
    return 2*m0 / ( gamma**.5 - 3 )

def sigma_hat(m0):
    gamma = m0*np.log(alpha) / b 
    return 2*m0 / ( gamma**.5 - 3 )


U = [J_1, J_2]


# variables
a = 10
b = 1
d = b

alpha = 3 


m0 = 36*b / np.log(alpha) * 1.5

case=0
if case == 0:
    sigma = abs(sigma_tilde(m0)) * .9
else:
    sigma = abs(sigma_hat(m0)) * 1.2

size = 1000
S_1 = gt.strategy_space([0.001, m0], discrete=False, size=size)
S_2 = gt.strategy_space([0.001, m0], discrete=False, size=size)


S = [S_1, S_2]

P = 2

G = gt.game(U, S, P)

NE = np.array( G.find_NE() )

BR = G.BR



# plot the best response

BR1, s2 = G.BR_mapping(0)
BR2, s1 = G.BR_mapping(1)

threshold = 70
dydx1 = np.gradient(BR1)/np.gradient(S_2.domain)
idx1 = np.where(np.abs( dydx1  ) >= threshold)[0]
for k in idx1:
    BR1[k] = np.nan


dydx2 = np.gradient(BR2)/np.gradient(S_1.domain)
idx2 = np.where(np.abs( dydx2  ) >= threshold)[0]
for k in idx2:
    BR2[k] = np.nan


plt.figure(1)
plt.clf()
plt.plot(BR1, S_2.domain, '-', label='Best Response Player 1: $m_1^*(m_2)$')
plt.plot(S_1.domain, BR2, '--', label='Best Response Player 2: $m_2^*(m_1)$')

plt.plot(NE[:,0], NE[:,1], 'o', label='Nash Equilibria')


plt.xlabel('$m_1$')
plt.ylabel('$m_2$')
plt.legend()
plt.show()

savefig('NE_case{}_ns.eps'.format(case))
savefig('NE_case{}_ns.pgf'.format(case))





