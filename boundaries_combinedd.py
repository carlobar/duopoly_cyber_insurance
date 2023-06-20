import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib import cm
import pdb


def savefig(title):
    dir_images = './images/'
    try:
        plt.savefig(dir_images + title, bbox_inches = 'tight')
    except:
        plt.savefig(title, bbox_inches = 'tight')

def sigma_tilde(m0):
    return 36*b*m0/(m0*np.log(alpha) - 36*b)

def sigma_hat(m0):
    Gamma = ( 9*b / (m0*np.log(alpha)) )** 0.5
    return m0 * Gamma / (1-2*Gamma)


def sigma_tilde_ns(m0):
    gamma = m0*np.log(alpha) / b * (5/3)
    return 2*m0 / ( gamma**.5 - 3 )

def sigma_hat_ns(m0):
    gamma = m0*np.log(alpha) / b 
    return 2*m0 / ( gamma**.5 - 3 )



# variables
a = 10
b = 1
d = b

alpha = 3 



m_a = (27/5) * b / np.log(alpha)
m_b = 9*b / np.log(alpha)
m_c = 36*b / np.log(alpha)

m0 = 36*b / np.log(alpha) * 100

h = 1000
delta = 0.01
x1 = np.logspace(np.log(m_a+delta), np.log(m0), num=h, base=np.exp(1) )
x2 = np.logspace(np.log(m_b+delta), np.log(m0), num=h, base=np.exp(1) )
x3 = np.logspace(np.log(m_c+delta), np.log(m0), num=h, base=np.exp(1) )


y1 = sigma_tilde_ns(x1)
y2 = sigma_hat_ns(x2)
y3 = sigma_tilde(x3)
y4 = sigma_hat(x3)




hh = 1000
x0_b = np.logspace(np.log(delta), np.log(m0), num=hh, base=np.exp(1) )
x1_b = np.logspace(np.log(m_a+delta), np.log(m0), num=hh, base=np.exp(1) )
x2_b = np.logspace(np.log(m_b+delta), np.log(m0), num=hh, base=np.exp(1) )
x3_b = np.logspace(np.log(m_c+delta), np.log(m0), num=hh, base=np.exp(1) )

y1_b = sigma_tilde_ns(x1_b)
y2_b = sigma_hat_ns(x2_b)
y3_b = sigma_tilde(x3_b)
y4_b = sigma_hat(x3_b)



y_max = np.max(y1)

plt.figure(1)
plt.clf()
plt.plot(x1, y1, '-', color='cornflowerblue', label=r'$\acute \sigma (m_0)$ non-sharing')
plt.plot(x2, y2, '--', color='cornflowerblue', label=r'$\breve \sigma (m_0)$ non-sharing')
plt.plot(x3, y3, '-', color='forestgreen', label=r'$\tilde \sigma (m_0)$ sharing')
plt.plot(x3, y4, '--', color='forestgreen', label=r'$\hat \sigma (m_0)$ sharing')
plt.xlim([2, m0])
plt.ylim([10, y_max])
plt.yscale('log')
plt.xscale('log')

plt.xlabel('$m_0$')
plt.ylabel('$\sigma$')

plt.xticks([m_a, m_b, m_c], [r'$\frac{27 b }{ 5 \log \alpha}$', r'$\frac{9 b }{ \log \alpha}$', r'$\frac{36 b }{ \log \alpha}$'])
plt.yticks([], [])



##############################################
y_b = sigma_hat_ns(x3_b)

plt.fill_between(x2_b, y_max, y2_b, where=( x2_b < m_c ),facecolor="lightsteelblue", alpha=0.2, interpolate=True)

plt.fill_between(x3_b, y_b, y3_b, where=( y_b < y3_b ),facecolor="lightsteelblue", alpha=0.2, interpolate=True)
################################################3

plt.fill_between(x3_b, y_max, y4_b, where=( y4_b < y_max ),facecolor="lightsteelblue", alpha=0.2, interpolate=True)

xx = (m_c+m_b)*.4
plt.text(xx, y_max/4, r"A", ha='left')
plt.text((m_c)*2, y_max/4, r"B", ha='left')


plt.legend()
plt.show()




savefig('eq_boundary_complete.eps')
savefig('eq_boundary_complete.pgf')

