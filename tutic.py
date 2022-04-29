# Explore the model in: Tutić, A. (2021). Stochastic evolutionary dynamics in the volunteer's dilemma, 
# The Journal of Mathematical Sociology pp. 1–20. https://doi.org/10.1080/0022250X.2021.1988946

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

# ----------------------

# fitness functions \pi^w_c(i, h) and \pi^w_d(i, h)

def pic_fnc(w, i, h):
    return 1 - w + w * (beta-gamma)

def pid_fnc(w, i, h):
    if h == 2:
        return 1 - w + w * (beta*i/(n-1))
    else: # h > 2
        return 1 - w + w*beta * (1 - hypergeom.pmf(h-1, n-1, n-i-1, h-1))


# transition matrix

def calc_P(w, h):

    # precalculate pi_c^w and pi_d^w for each i
    picwV = [np.nan] + [ pic_fnc(w, i, h) for i in range(1, n+1) ]  # nan for i=0 because focal is cooperator
    pidwV = [ pid_fnc(w, i, h) for i in range(n) ] + [np.nan]       # nan for i=n because focal is defector


    # construct the transition matrix

    # define states as the number of cooperators in the population,
    # which goes from i = 0, ..., n

    # P(i,j) is the probability of transition from i -> j cooperators
    P = list()

    # i = 0 is an absorbing state
    P.append([1] + [0]*n)

    # i = 1, ..., n-1 is tridiagonal
    for i in range(1, n):

        v = [0]*(n+1)
        v[i-1] =   (i/n)   * pidwV[i]*(n-i) / ( picwV[i]*i + pidwV[i]*(n-i) ) # probability lose one cooperator
        v[i+1] = ((n-i)/n) * picwV[i]*i     / ( picwV[i]*i + pidwV[i]*(n-i) ) # probability gain one cooperator
        v[i] = 1 - v[i-1] - v[i+1]                                            # probability stay unchanged
        P.append(v)

    # i = n is an absorbing state
    P.append([0]*n + [1])

    return np.array(P)


# ----------------------

# fixed parameters
# ---

n = 100             # population size
beta = 4            # benefit of public good
gamma = 2           # cost to volunteer
w_gridsize = 100    # how many w values to evaluate for the fixation probabilities plot plot


# plot the fixation probability versus selection strength
# ---

for h in [2, 5]: # Fig. 2 and 5 in Tutic 2021

    # for each selection strength w, find the probability of absorption to all-cooperator state

    # a place to store fixation probabilities
    p_allcoop_i0loV = list()
    p_allcoop_i0hiV = list()

    wV = np.linspace(0, 1, w_gridsize)
    for w in wV:

        # construct the transition matrix
        P = calc_P(w, h)

        # put it in canonical form
        absorbing_idxs = [ i for i in range(n+1) if P[i,i] == 1 ]
        transient_idxs = [ i for i in range(n+1) if i not in absorbing_idxs ]
        R = P[transient_idxs, :][:, absorbing_idxs]
        Q = P[transient_idxs, :][:, transient_idxs]

        # find the probability of absorbtion to i = n (all cooperators) from intial i0_lo = 1 and i0_hi = n-1

        # the fundamental matrix, N = (I-Q)^{-1}
        N = np.linalg.inv(np.identity(len(transient_idxs)) - Q)

        # element (i,j) is the probability of ending in ith given it started in jth state
        absorbing_probs = N @ R 

        # extract fixation probabilities for i0 = 1 (row 0) and i0 = n-1 (row n-2)
        p_allcoop_i0lo = absorbing_probs[0, 1]      # column 1 is fixation to all-cooperator state
        p_allcoop_i0hi = absorbing_probs[n-2, 1]

        # store them
        p_allcoop_i0loV.append(p_allcoop_i0lo)
        p_allcoop_i0hiV.append(p_allcoop_i0hi)


    # plot fixation probs
    plt.rcParams["figure.figsize"] = (4,3)
    plt.plot(wV, p_allcoop_i0loV, label=r'$i_0 = 1$')
    plt.plot(wV, p_allcoop_i0hiV, label=r'$i_0 = 99$')
    plt.xlabel(r'selection pressure $w$')
    plt.ylabel(r'fixation probabilities')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('tutic_fixnprob_h' + str(h) + '.pdf')
    plt.close()


# plot the fixation probability versus initial i for w=0.8 and h=5, 
# ---

h = 5
w = 0.8

# construct the transition matrix
P = calc_P(w, h)

# put it in canonical form
absorbing_idxs = [ i for i in range(n+1) if P[i,i] == 1 ]
transient_idxs = [ i for i in range(n+1) if i not in absorbing_idxs ]
R = P[transient_idxs, :][:, absorbing_idxs]
Q = P[transient_idxs, :][:, transient_idxs]

# find the probability of absorbtion to i = n (all cooperators) from intial i0_lo = 1 and i0_hi = n-1

# the fundamental matrix, N = (I-Q)^{-1}
N = np.linalg.inv(np.identity(len(transient_idxs)) - Q)

# element (i,j) is the probability of ending in ith given it started in jth state
absorbing_probs = N @ R 

# extract fixation probabilities for i0 = 1 (row 0) and i0 = n-1 (row n-2)
p_allcoop_i0lo = absorbing_probs[0, 1]      # column 1 is fixation to all-cooperator state
p_allcoop_i0hi = absorbing_probs[n-2, 1]

# plot fixation probs
plt.rcParams["figure.figsize"] = (4,3)
plt.plot(range(1, n), absorbing_probs[:, 1])
plt.xlabel(r'initial cooperators $i_0$')
plt.ylabel(r'fixation probabilities')
plt.ylim((-0.05, 1.05))
plt.tight_layout()
plt.savefig('tutic_fixnprob_h' + str(h) + '_w' + str(int(w*10)) + '.pdf')
plt.close()


# plot the time to fixation when h = 5 and w=0.8
# ---

# construct the transition matrix
P = calc_P(w, h)

# put it in canonical form
absorbing_idxs = [ i for i in range(n+1) if P[i,i] == 1 ]
transient_idxs = [ i for i in range(n+1) if i not in absorbing_idxs ]
R = P[transient_idxs, :][:, absorbing_idxs]
Q = P[transient_idxs, :][:, transient_idxs]

# the fundamental matrix, N = (I-Q)^{-1}
N = np.linalg.inv(np.identity(len(transient_idxs)) - Q)

# the expected length of time in the transient states given it started in jth state 
# is the sum of the jth row of N
transient_time = np.sum(N, axis=1)

# plot it
plt.rcParams["figure.figsize"] = (4,3)
plt.plot(range(1, n), transient_time)
plt.xlabel(r'initial number of cooperators $i_0$')
plt.ylabel(r'time before fixation')
plt.tight_layout()
plt.savefig('tutic_transient_time.pdf')
plt.close()


# plot the quasi-stationary distribution
# ---

# mu are eigenvalues and u are left eigenvectors
muV, uV = np.linalg.eig(np.transpose(Q)) # transpose it so it returns left eigenvector

# find leading eigenvalue
idxs = np.argsort(muV)
mu1 = muV[idxs[-1]]

# stationary distribution is the normalised leading left eigenvector
u = uV[:, idxs[-1]]
stationary_distn = u/sum(u)

# expected value to compare to Tutic (2021) Fig. 6
expected = sum( prob*i for i, prob in zip(range(1, n+1), stationary_distn) )

# does it have quasi stationary behaviour?
mu2 = muV[idxs[-2]]         # second largest eigenvalue
gamma = mu1-np.real(mu2)    # spectral gap (see van Doorn & Pollett (2013))
'''
mu1   = 0.9999989308081333
mu2   = 0.9945178632790972
1-mu1 = 1.0691918667093958e-06
gamma = 0.005481067529036099  # gamma >> 1-mu1 indicates quasi-stationary behaviour (van Doorn & Pollett 2013)
'''

# plot it
plt.rcParams["figure.figsize"] = (4,3)
plt.bar(range(1, n), stationary_distn, alpha=0.7)
plt.axvline(expected, color='blue')
plt.xlabel(r'number of cooperators')
plt.ylabel(r'quasi-stationary distribution')
plt.tight_layout()
plt.savefig('tutic_qss.pdf')
plt.close()
