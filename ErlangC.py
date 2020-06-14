import numpy as np

def ErlangC(lm, mu, n): 
    
    N = int(max(lm/mu, n) * 100)
    pi = np.ones(N + 1)
    for i in range(N):
        pi[i+1] = pi[i] * lm / (min(i+1, n) * mu)
    
    C=sum(pi)
    pi = pi / C
    q = [max(0, i - n) for i in range(N + 1)]
    EQ = np.dot(pi, q)
    return EQ