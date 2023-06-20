import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import sys
'''
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
'''

def savefig(title):
    dir_images = '../manuscript/images/'
    try:
        plt.savefig(dir_images + title, bbox_inches = 'tight')
    except:
        plt.savefig(title, bbox_inches = 'tight')


n=10000000


# variables for the price function
a = 10
b = 1
d = b

# variables for the noise
sigma = 4
m1 = 2
m2 = 2

# random variables of cost and noise
cost = stats.norm(0, (sigma)**.5)
e1 = stats.norm(0, (m1)**.5)
e2 = stats.norm(0, (m2)**.5)

# samples of the cost with noise
cost_sample = cost.rvs(size=n)
Z1_sample = cost_sample + e1.rvs(size=n)
Z2_sample = cost_sample + e2.rvs(size=n)

# actual observations
z1 = 3
z2 = 3



# plot the distributions of costs and signals
plt.figure(1)
plt.clf()
sns.kdeplot(cost_sample, label='$P[C=x]$')
sns.kdeplot(Z1_sample, label='$P[Z_1=x]$')
sns.kdeplot(Z2_sample, label='$P[Z_2=x]$')
plt.title('Distribution of cost $C$ and private signals $Z_1$ and $Z_2$')
plt.xlabel('x')
plt.legend()
plt.show()


# calculation of the cost distribution conditional to the observations
hat_mu_1 = sigma/(sigma+m1) * z1
hat_sigma_1 = sigma/(sigma+m1) * m1

hat_mu_2 = sigma/(sigma+m2) * z2
hat_sigma_2 = sigma/(sigma+m2) * m2

k0 = sigma*(m1+m2) + m1*m2
hat_mu = (z1*m2 + z2*m1)*sigma/k0
hat_sigma = sigma*m1*m2/k0


hat_C_1 = stats.norm(hat_mu_1, (hat_sigma_1)**.5)
hat_C_2 = stats.norm(hat_mu_2, (hat_sigma_2)**.5)
hat_C = stats.norm(hat_mu, (hat_sigma)**.5)


plt.figure(2)
plt.clf()
sns.kdeplot(cost.rvs(size=n), label='$P[C=x]$')
sns.kdeplot(hat_C_1.rvs(size=n), label='$P[C=x| Z_1]$')
#sns.kdeplot(hat_C_2.rvs(size=n), label='$P[C=x| Z_2]$')
sns.kdeplot(hat_C.rvs(size=n), label='$P[C=x| Z_1, Z_2]$')
plt.title('Distribution of the cost estimation given $Z_1$={} and $Z_2$={}'.format(z1, z2))
plt.xlabel('x')
plt.legend()
plt.grid()
plt.show()

savefig('cost_dist.eps')
savefig('cost_dist.pgf')


plt.figure(3)
plt.clf()
sns.kdeplot(x=Z1_sample, y=Z2_sample)
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')
plt.title('Bivariate distribution of $Z_1$ and $Z_2$')
plt.show()

savefig('bivariate_dist.eps')
savefig('bivariate_dist.pgf')









