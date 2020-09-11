
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_spd_matrix
from numpy.linalg import eig
from numpy.random import multivariate_normal as mv_normal

n = 2 # working in R^2

# cov matrix is always symmetric, SPD
A = make_spd_matrix(n) # generating random 2x2 SPD matrix
# X is a normalized matrix of eigenvectors, l is an array of corresponding eigenvalues
l, X = eig(A)

# Want to create nxn matrix diag(l)
Lambda = np.zeros((n,n), float) # diagonal elements are eigenvalues, non-diagonal elements are 0
np.fill_diagonal(Lambda, l) # populating diagonal entries

B = np.matmul(X,Lambda)
B = np.matmul(B, X.T) # eigen decomposition: A = X*Lambda*X.T

print(A) # printing covariance matrix

num_samples = 500

# creating vector mu with random means for x1 and x2
mu = [random.randint(1,101) for i in range(0,n)] # arbitrarily selecting means between 1 and 100
print('mu: {}'.format(mu))

X = [None]*n # would also work for n-dimensional data

# creating 2D scatter plot for x1, x2
X = mv_normal(mu, A, num_samples).T
plt.plot(X[0],X[1], 'o')
plt.show()










