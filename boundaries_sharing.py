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
        #pdb.set_trace()
        plt.savefig(title, bbox_inches = 'tight')

def sigma_tilde(m0):
    return 36*b*m0/(m0*np.log(alpha) - 36*b)

def sigma_hat(m0):
    Gamma = ( 9*b / (m0*np.log(alpha)) )** 0.5
    return m0 * Gamma / (1-2*Gamma)


# variables
a = 10
b = 1
d = b

alpha = 3 

#m0 = 36*b / np.log(alpha) * 1.2
m0 = 36*b / np.log(alpha) 

m1 = np.linspace(0, m0, 200 )
m2 = np.linspace(m0+0.1, m0*3, 200 )

y1 = sigma_hat(m2)
y2 = sigma_tilde(m2)



plt.figure(1)
plt.clf()

plt.plot(m2, y1, color='cornflowerblue', label='$\hat \sigma(m_0)$')
plt.plot(m2, y2, '--', color='forestgreen', label=r'$\tilde \sigma(m_0)$')

plt.text(m0/2, np.max(y1)/3, r"$(m_0, m_0)$", ha='center')
plt.text(m0*2.7/2, np.max(y1)/3, r"$(m, m_0)$", ha='left')
plt.text(m0*2.5/2, np.min(y2)*.9, r"$(m_0, m_0)$", ha='left')

plt.xlim([0, m0*3])
plt.yscale('log')
plt.legend()
plt.xlabel('$m_0$')
plt.ylabel('$\sigma$')

plt.xticks([m0], [r'$\frac{36b}{ \log (\alpha)}$'])
plt.yticks([], [])
plt.title('Possible equilibria sharing information')
plt.show()

savefig('eq_boundary_sharing.eps')
savefig('eq_boundary_sharing.pgf')



