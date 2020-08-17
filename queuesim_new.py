import random
import numpy as np
import matplotlib.pyplot as plt

def ErlangC_speedup(lm, mu, n, x): 
    
    N = int(max(lm/mu, n) * 100)
    pi = np.ones(N + 1)
    for i in range(N):
        if i >= x-1:
            pi[i+1] = pi[i] * lm / (min(i+1, n) * 1.5 * mu)
        else:
            pi[i+1] = pi[i] * lm / (min(i+1, n) * mu)
    
    C=sum(pi)
    pi = pi / C
    q = [max(0, i - n) for i in range(N + 1)]
    EQ = np.dot(pi, q)
    return EQ

def pi_speedup(lm, mu, n, x): 
    
    N = int(max(lm/mu, n) * 100)
    pi = np.ones(N + 1)
    for i in range(N):
        if i >= x-1:
            pi[i+1] = pi[i] * lm / (min(i+1, n) * 1.5 * mu)
        else:
            pi[i+1] = pi[i] * lm / (min(i+1, n) * mu)
    
    C=sum(pi)
    pi = pi / C
    P = sum(pi[x:])
    return P

def generateIA(lm):
  #Poisson Arrivals
  ia = -1*np.log(random.random())/lm

  #sinusoidal inter-arrival times
  #lm = lm*1.15
  #ia = -1*np.log(random.random())/lm

  return ia

def generateST(mu):
  #Exponential Service Times
  st = -1*np.log(random.random())/mu

  #two-point mu
  #st = 0.5/mu
  #st2 = 1.5/mu
  #if random.random() > 0.5:
  #  st = st2

  #lognormal distribution with mu=1 and var=0.25
  #ln_sig = 0.4723807271
  #ln_mu = np.log(mu)-(ln_sig**2)/2
  #st = random.lognormvariate(ln_mu, ln_sig)

  return st

def run_sim(K, T, lm, mu, n):
  tp = 0
  mu_bar = 1.5*mu
  R = np.array([1e30]*n)
  X = 0
  A = generateIA(lm)
  dlt = min(A, R[0])
  t = tp + dlt

  EQT = 0 #cumulative customer waiting time statistic
  PT = 0 #cumulative time spent past threshold statistic
  abandonments = 0 #cumulative abandonments count statistic

  while t <= T:
    EQT += max(X-n, 0) * dlt
    if X >= K:
      PT += dlt

    if A < min(R):
      if X >= K:
        R = R - dlt*(mu_bar/mu)
      else:
        R = R - dlt
      A = generateIA(lm)
      
      #for non-stationary arrivals: uncomment below and indent the follwing if and else statements
      #lmt = lm*(1+0.15*np.sin(t*2*np.pi/10))
      #if random.random() <= lmt/115:
      
      if X < n or (random.random() > 0.1):
        X = X + 1
        if X <= n:
          R[-1] = generateST(mu)
      else:
        abandonments += 1
      
      R = np.sort(R)
    
    elif min(R) < A:
      if X >= K:
        R = R - dlt*(mu_bar/mu)
      else:
        R = R - dlt
      A = A - dlt
      X = X - 1
      if X >= n:
        R[0] = generateST(mu)
      else:
        R[0] = 1e30
      R = np.sort(R)
    tp = t
    if X >= K:
      dlt = min(A, mu/mu_bar*R[0])
    else:
      dlt = min(A, R[0])
    t = tp + dlt
  
  return EQT/T, PT/T, abandonments/T

#
results = []
for K in range(70, 130, 5):
  print(K, '/', '125')
  results.append(run_sim(K, 5e4, 100, 1, 95))

EQs = [res[0] for res in results]
Ps = [res[1] for res in results]
As = [res[2] for res in results]

c = 5
h = 15
t = 1000
a = 28

emp_costs = [c * 95 + h * EQs[i] + t * Ps[i] + a * As[i] for i in range(len(Ps))] 
theo_costs = [c * 95 + h * ErlangC_speedup(100, 1, 95, K) + t * pi_speedup(100, 1, 95, K) for K in range(70, 130, 5)] 

plt.plot(range(70, 130, 5), theo_costs)
plt.plot(range(70, 130, 5), emp_costs, linestyle='--')
plt.grid(True)
plt.xlabel('Threshold', fontsize=15)
plt.ylabel('Cost', fontsize=15)
plt.title('Empirical vs. Theoretical Costs', fontsize = 15)
plt.legend(['Theoretical', 'True'], fontsize=12)
plt.show()