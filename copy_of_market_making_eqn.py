*Market Making*
"""
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from scipy.sparse import diags
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline

np.set_printoptions(linewidth=200)

def compute_analytic( lamda, kappa, phi, alpha, q_max, q_min, T):
    
    n = q_max - q_min + 1 # size of our ODE system
    
    z = np.array( [ np.exp(-alpha * kappa * i**2)  for i in range(q_max, q_min-1, -1)] )
    
    diagonals = [
        [ -phi * kappa * i**2 for i in range(q_max, q_min-1, -1) ], 
        [lamda / np.e] * (n-1),
        [lamda / np.e] * (n-1),    
    ]
    
    M = diags(diagonals, [0,1,-1])
    
    ts = np.linspace(0, 30, 100)

    G = np.zeros((n, len(ts)))

    for idx, t in enumerate(ts):
        A = M * (T-t)
        a, v = LA.eigh(A.todense()) #eigen decomposition.    
        w_tq = v.dot( diags(np.exp(a)).todense().dot(v.T)).dot(z)
        g_tq = np.log(w_tq) / kappa
        G[:,idx] = g_tq
    
    del_plus = np.zeros(((n-1), len(ts)))

    f, (ax0, ax1) = plt.subplots(1,2,figsize=(14,6))

    q_map = dict( (q, i) for i, q in enumerate(range(q_max, q_min-1, -1)))

    lookup = lambda q : q_map[q] 

    ax0.set_ylabel("Sell Depth ($\delta^+$)")
    ax0.set_xlabel("Time (secs)")
    ax1.set_ylabel("Buy Depth ($\delta^-$)")
    ax1.set_xlabel("Time (secs)")

    def get_color( q ):
        return ((q-q_min) / float( q_max - q_min))

    for q in range(q_max, q_min, -1):
        del_plus[lookup(q)] = ( G[lookup(q)] - G[lookup(q-1)] ) + ( 1. / kappa )
        a = 0.3 if q !=2 else 1.
        ax0.plot(ts, del_plus[lookup(q)],label = 'q = %s' %q, lw='4', color=plt.cm.Paired(get_color(q)), alpha=a )

    for q in range(q_min, q_max ):
        del_plus[lookup(q)-1] = ( G[lookup(q)] - G[lookup(q+1)] ) + ( 1. / kappa )
        a = 0.3 if q !=2 else 1.
        ax1.plot(ts, del_plus[lookup(q)-1], label = 'q = %s' %q, lw='4', color=plt.cm.Paired(get_color(q)), alpha=a)    

    ax0.legend()
    ax1.legend()

    title = '$\kappa^+=\kappa^+={},\lambda^+=\lambda^-={},\phi={},\\alpha={},T={}$'
    title = title.format(kappa, lamda, phi, alpha, T)

    f.suptitle(title)

    plt.show()



lamda = 1.0    # arrival rate
kappa = 100.0  # fill rate constant
phi = 0.0001   # running inventory penalty. try 0.001 to see -ve sell depth
alpha = 0.0001 # penalty for terminal liquidation.
T = 30.0       # terminal time.

q_max = 3      # max inventory level
q_min = -3     # min inventory level

compute_analytic( lamda, kappa, phi, alpha, q_max, q_min, T)

def compute_finite_difference( lamda_p, lamda_m, kappa_p, kappa_m, alpha, q_max, q_min, T):
    M = 100
    dt = T / M

    C1 = lamda_p / (np.e * kappa_p)
    C2 = lamda_m / (np.e * kappa_m)

    q_max = 3      # max inventory level
    q_min = -3     # min inventory level

    n = q_max - q_min + 1 # size of our ODE system

    # terminal conditions.
    z = np.array([ -alpha * i**2 for i in range(q_max, q_min-1, -1)])
    
    q_map = dict( (q, i) for i, q in enumerate(range(q_max, q_min-1, -1)))
    lookup = lambda q : q_map[q] 
        
    G = np.zeros((n, M+1))
    G[:,-1] = z
    
    for idx in range(M, 0, -1):

        gm_prev = np.zeros(n)
        gm = G[:,idx]
        for q in range(q_max, q_min-1, -1):
            if q == q_max:
                gm_prev[lookup(q)] = gm[lookup(q)] + (- (phi*q**2) 
                                                      + C1 * np.exp( kappa_p*( gm[lookup(q-1)]-gm[lookup(q)] ) ) ) * dt
            elif q == q_min:
                gm_prev[lookup(q)] = gm[lookup(q)] + (- (phi*q**2) 
                                                      + C2 * np.exp( kappa_m*( gm[lookup(q+1)]-gm[lookup(q)] ) ) ) * dt
            else:
                gm_prev[lookup(q)] = gm[lookup(q)] + (- (phi*q**2) 
                                      + C1 * np.exp(kappa_p*(gm[lookup(q-1)]-gm[lookup(q)])) 
                                      + C2 * np.exp(kappa_m*(gm[lookup(q+1)]-gm[lookup(q)])))*dt

        G[:,idx-1] = gm_prev
    
    ts = np.linspace(0, T, M+1)
    
    del_plus = np.zeros_like(G)

    f, (ax0, ax1) = plt.subplots(1,2,figsize=(14,6))
    
    def get_color( q ):
        return ((q-q_min) / float( q_max - q_min))
    
    ax0.set_ylabel("Sell Depth ($\delta^+$)")
    ax0.set_xlabel("Time (secs)")
    ax1.set_ylabel("Buy Depth ($\delta^-$)")
    ax1.set_xlabel("Time (secs)")

    for q in range(q_max, q_min, -1):
        del_plus[lookup(q)] = ( G[lookup(q)] - G[lookup(q-1)] ) + ( 1. / kappa )
        a = 0.3 if q !=2 else 1.
        ax0.plot(ts, del_plus[lookup(q)], label = 'q = %s' %q, lw='4', color=plt.cm.Paired(get_color(q)), alpha=a)

    for q in range(q_min, q_max ):
        del_plus[lookup(q)-1] = ( G[lookup(q)] - G[lookup(q+1)] ) + ( 1. / kappa )
        a = 0.3 if q !=2 else 1.
        ax1.plot(ts, del_plus[lookup(q)-1], label = 'q = %s' %q, lw='4', color=plt.cm.Paired(get_color(q)), alpha=a)        

    ax0.legend()
    ax1.legend()

    title1 = '$\kappa^+=\kappa^+={},\lambda^+=\lambda^-={},\phi={},\\alpha={},T={}$'
    title = title1.format(kappa, lamda, phi, alpha, T)

    f.suptitle(title)

    plt.show()

lamda_p = lamda_m = 1.0 # arrival rate
kappa_p = kappa_m = 100. # fill rate constant
phi = 0.0001   # running inventory penalty. try 0.001 to see -ve sell depth
alpha = 0.0001 # penalty for terminal liquidation.
q_max = 3      # max inventory level
q_min = -3     # min inventory level
T = 30.0       # terminal time.
compute_finite_difference( lamda_p, lamda_m, kappa_p, kappa_m, alpha, q_max, q_min, T)

lamda_p = 0.75 # buy arrival rate
lamda_m = 1.1  # sell arrival rate
kappa_p = kappa_m = 100. # fill rate constant
phi = 0.0001   # running inventory penalty. try 0.001 to see -ve sell depth
alpha = 0.0001 # penalty for terminal liquidation.
q_max = 3      # max inventory level
q_min = -3     # min inventory level
T = 30.0       # terminal time.
compute_finite_difference( lamda_p, lamda_m, kappa_p, kappa_m, alpha, q_max, q_min, T)

lamda_p = 1.0 # buy arrival rate
lamda_m = 1.0  # sell arrival rate
kappa_p = 150.
kappa_m = 50. # fill rate constant
phi = 0.0001   # running inventory penalty. try 0.001 to see -ve sell depth
alpha = 0.0001 # penalty for terminal liquidation.
q_max = 3      # max inventory level
q_min = -3     # min inventory level
T = 30.0       # terminal time.
compute_finite_difference( lamda_p, lamda_m, kappa_p, kappa_m, alpha, q_max, q_min, T)

lamda_p = 1.0 # buy arrival rate
lamda_m = 1.0  # sell arrival rate
kappa_p = 150.
kappa_m = 50. # fill rate constant
phi = 0.0001   # running inventory penalty. try 0.001 to see -ve sell depth
alpha = 0.001 # penalty for terminal liquidation.
q_max = 3      # max inventory level
q_min = -3     # min inventory level
T = 30.0       # terminal time.
compute_finite_difference( lamda_p, lamda_m, kappa_p, kappa_m, alpha, q_max, q_min, T)

lamda_p = 1.0 # buy arrival rate
lamda_m = 1.0  # sell arrival rate
kappa_p = 150.
kappa_m = 50. # fill rate constant
phi = 0.0005   # running inventory penalty. try 0.001 to see -ve sell depth
alpha = 0.0001 # penalty for terminal liquidation.
q_max = 3      # max inventory level
q_min = -3     # min inventory level
T = 30.0       # terminal time.
compute_finite_difference( lamda_p, lamda_m, kappa_p, kappa_m, alpha, q_max, q_min, T)
