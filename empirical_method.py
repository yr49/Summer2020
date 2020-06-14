from ErlangC import ErlangC
import matplotlib.pyplot as plt
from run_experiment import run_experiment

N = range(101, 126, 5)
T = [10, 100, 1000, 10000]
c = 5
h = 15
costs = []
for t in T:
  print()
  print("t = ", t)
  cost = []
  for n in N:
    print("n = ", n)
    lm, mu, EQ = run_experiment(n, t)
    cost.append(c*n + h*EQ)
  costs.append(cost)
plt.yscale('log')
for cost in costs:
  plt.plot(N, cost)
  
#lm = 10
#mu = 1
T = [10, 100, 1000, 10000]
costs = []
for t in T:
  lm, mu, Q = run_experiment(101, t)
  c = 5
  h = 15
  x = range(101, 126)
  cost = []
  for n in x:
    EQ = ErlangC(lm, mu, n)
    cost.append(c*n + h*EQ)
  costs.append(cost)
plt.yscale('log')
for cost in costs:
  plt.plot(x, cost)

lm = 100
mu = 1
true = []
for n in x:
  EQ = ErlangC(lm, mu, n)
  true.append(c*n + h*EQ)
plt.plot(x, true)

plt.legend(['T=10', 'T=100', 'T=1,000', 'T=10,000', 'True Cost'])
plt.grid(True)
plt.xlabel("Staffing Level", fontsize=15)
plt.ylabel("Cost", fontsize=15)
plt.title("Empirical Approximation of Cost: $T$ long simulation", fontsize=15)
plt.show()