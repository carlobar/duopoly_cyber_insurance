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
    gamma = m0*np.log(alpha) / b * (5/3)
    return 2*m0 / ( gamma**.5 - 3 )

def sigma_hat(m0):
    gamma = m0*np.log(alpha) / b 
    return 2*m0 / ( gamma**.5 - 3 )


# variables
a = 10
b = 1
d = b

alpha = 3 



m0_a = (27/5) * b / np.log(alpha)
m0_b = 9*b / np.log(alpha)

m0_1 = 12*b/(5*np.log(alpha))


m0 = m0_b * 1.5
m = np.linspace(0, m0, 200 )


m_a = np.linspace(m0_a+0.05, m0, 200 )
m_b = np.linspace(m0_b+0.05, m0, 200 )

m1 = np.linspace(0, m0_1, 200 )
m2 = np.linspace(m0_1, m0_a, 200 )


y_a = sigma_tilde(m_a)
y_b = sigma_hat(m_b)



def f1(x1, x2, sigma):
    return (4*x1*x2 + sigma**2) / (4*x1*x2 - sigma**2)

def f2(x1, x2, sigma):
    return (2*x2 - sigma)**2 / (4*x1*x2 - sigma**2)**2

def marginal_cost(m):
    return - 1 / ( m*np.log(alpha) )

def J_dot(m1, m2, sigma):
    x1 = sigma + m1
    x2 = sigma + m2
    return  -sigma**2/b * f1(x1, x2, sigma) * f2(x1, x2, sigma) - marginal_cost(m1)

def J_dot_upp(m1, m2, sigma):
    x = sigma+m0
    return -sigma**2/b * 1/(2*x+sigma)**2 - marginal_cost(m0)


max_y = np.max(y_a)




plt.figure(1)
plt.clf()

plt.plot(m_a, y_a, color='cornflowerblue', label=r'$\acute{ \sigma}(m_0)$')
plt.plot(m_b, y_b, '--', color='forestgreen', label=r'$\breve{\sigma}(m_0)$')

plt.text(m0_b*.65, np.min(y_a), r"$(m_0, m_0)$", ha='left')
plt.text((m0+m0_b)/2*.95, max_y/4 , r"$(m, m_0) $" "\n" "$ (m, m) $", ha='left')
plt.text((m0_1+m0_a)/2, max_y*.5, r"$(m_0, m_0)$", ha='center')


plt.xlim([0, m0])
ylim_0 = plt.ylim()
plt.ylim([np.min(y_a)/2, max_y])
plt.yscale('log')
plt.legend()
plt.xlabel('$m_0$')
plt.ylabel('$\sigma$')

plt.xticks([m0_a, m0_b], [r'$\frac{27 b }{ 5 \log (\alpha)}$', r'$\frac{9 b }{ \log (\alpha)}$'])
plt.yticks([], [])
plt.title('Possible equilibria holding information')
plt.show()

savefig('eq_boundary_non-sharing.eps')
savefig('eq_boundary_non-sharing.pgf')



