from run_experiment import run_experiment
from ErlangC import ErlangC
import numpy as np

x1, x2, x3, x4, x5 = range(101, 126), range(101, 126, 2), range(101, 126, 3), range(101, 126, 4), range(101, 126, 5)
xs = [x1, x2, x3, x4, x5]
c = 5.
h = 15.

dictn = {}
for n in x1:
  for T in [10, 100, 1000, 10000]:
	dictn[(n, T)] = []	

dictn2 = {}
for N in [len(x) for x in xs]:
  for T in [10, 100, 1000, 10000]:
    dictn2[(N, T)] = []
	
for i in range(100):
  for n in x1:
    #print("Simulating n = ", n)
    lm, mu, EQ = run_experiment(n, 10000, dictn)
  print("So far ", len(dictn[(101, 10000)]), " experiements have been run")
  f = open("/content/drive/My Drive/Summer Research 2020/dictn.txt","w")
  f.write( str(dictn) )
  f.close()
  
for T in [10, 100, 1000, 10000]:
  for i in range(100):
    cost = [c*n+h*dictn[(n, T)][i] for n in x1]
    for x in xs:
      y = [cost[i-101] for i in x]
      mymodel = np.poly1d(np.polyfit(x, y, 3))
      cost_pred = mymodel(x1)
      min_i = np.argmin(cost_pred)
      min_n = x1[min_i]
      dictn2[(len(x), T)].append(min_n)

optimal_n = 112
optimal_cost = c*optimal_n + h*ErlangC(100, 1, optimal_n)
for key in dictn2:
  arr = dictn2[key]
  N, T = key[0], key[1]
  percent_wrong = 0
  avg_percent_optimality_gap = 0
  for n in arr:
    if n != optimal_n:
      percent_wrong += 1
      avg_percent_optimality_gap += np.absolute(c*n + h*ErlangC(100, 1, n) - optimal_cost) / (100 * optimal_cost)
  print("For N, T pair ", N, ", ", T, " percent wrong is ", percent_wrong, ", avg percent optimality gap is", avg_percent_optimality_gap)