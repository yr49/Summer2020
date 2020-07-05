import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from run_experiment import run_experiment
from ErlangC import ErlangC
from sympy import solve, im
from sympy.abc import x

N = range(101, 131)
T = [10, 100, 1000, 10000]
c = 5
h = 15
b = 120
Xs = []
for t in T:
  print()
  print("t = ", t)
  X = []
  for n in N:
    print("n = ", n)
    lm, mu, a, EQ, alpha = run_experiment(n, t)
    X.append([EQ, alpha])
  Xs.append(X)

def method1(EX, n):
  rhos = solve(EX-n-(1+x**2)/(2*(1-x)))
  for rho in rhos:
    if im(rho) == 0 and rho >= 0: return rho*n

def method2(EX, n, alpha):
  rhos = solve(EX-n*x-(x/(1-x))*alpha)
  for rho in rhos:
    if im(rho) == 0 and rho >= 0: return rho*n

def method3(EX, n):
  return EX

#either import an existing dataframe and X value arrays
df = pd.read_csv('/content/drive/My Drive/Summer Research 2020/rho-fitting-lognormal.csv')
N=range(101, 131)
Xs= [[[137.0, 1.0], [91.0, 0.0], [96.0, 0.0], [105.0, 1.0], [149.0, 1.0], [100.0, 0.0], [98.0, 0.0], [101.0, 0.0], [104.0, 0.0], [94.0, 0.0], [95.0, 0.0], [90.0, 0.0], [98.0, 0.0], [92.0, 0.0], [107.0, 0.0], [91.0, 0.0], [96.0, 0.0], [91.0, 0.0], [119.0, 1.0], [109.0, 0.0], [104.0, 0.0], [101.0, 0.0], [94.0, 0.0], [90.0, 0.0], [98.0, 0.0], [114.0, 0.0], [114.0, 0.0], [81.0, 0.0], [110.0, 0.0], [94.0, 0.0]],
 [[123.5, 0.8], [130.3, 0.9], [107.2, 0.6], [101.9, 0.5], [113.3, 0.7], [112.9, 0.7], [107.5, 0.5], [97.9, 0.2], [99.4, 0.2], [103.0, 0.2], [95.9, 0.0], [94.8, 0.0], [99.5, 0.0], [96.3, 0.0], [102.2, 0.0], [104.3, 0.2], [99.0, 0.1], [97.7, 0.0], [100.1, 0.0], [102.3, 0.0], [98.9, 0.0], [102.6, 0.0], [104.0, 0.0], [103.6, 0.1], [100.8, 0.0], [101.9, 0.0], [102.1, 0.0], [100.2, 0.0], [103.2, 0.1], [94.3, 0.0]],
 [[144.76, 0.85], [116.83, 0.71], [113.05, 0.67], [108.5, 0.58], [105.34, 0.49], [102.59, 0.33], [105.16, 0.44], [102.25, 0.33], [99.56, 0.16], [102.92, 0.23], [100.9, 0.16], [100.25, 0.16], [99.2, 0.11], [100.87, 0.12], [101.3, 0.12], [100.82, 0.06], [100.17, 0.06], [100.97, 0.06], [98.54, 0.05], [100.89, 0.05], [101.35, 0.05], [101.02, 0.01], [102.5, 0.0], [99.16, 0.02], [100.71, 0.0], [99.57, 0.0], [99.52, 0.01], [101.07, 0.0], [99.25, 0.01], [99.07, 0.0]],
 [[148.53, 0.873], [121.027, 0.763], [113.983, 0.649], [109.226, 0.574], [105.827, 0.474], [104.596, 0.411], [102.53, 0.342], [102.46, 0.335], [102.562, 0.278], [101.619, 0.223], [101.217, 0.199], [100.998, 0.169], [100.577, 0.12], [101.348, 0.134], [100.11, 0.084], [99.623, 0.059], [99.893, 0.06], [100.039, 0.045], [100.018, 0.04], [100.026, 0.032], [99.714, 0.026], [100.282, 0.018], [100.358, 0.019], [100.393, 0.013], [100.184, 0.014], [100.039, 0.01], [100.152, 0.004], [99.908, 0.008], [99.618, 0.004], [99.898, 0.005]]]

#or create new dataframe and save it
df = pd.DataFrame(columns = ['n', 'EX', 'T', 'method1', 'method2', 'method3'])
for i in range(4):
  T = [10, 100, 1000, 10000][i]
  for j in range(len(Xs[i])):
    EX = Xs[i][j][0]
    alpha = Xs[i][j][1]
    n = N[j]
    df.loc[i*len(Xs[i])+j]=[n, EX, T, method1(EX, n), method2(EX, n, alpha), method3(EX, n)]
	
df.to_csv('/content/drive/My Drive/Summer Research 2020/rho-fitting-blocking.csv', index=False)

As = []
for X in Xs:
  A = []
  for pair in X:
    A.append(pair[1])
  As.append(A)
  
def plot_rhos():
	N=range(101, 131)
	Ts = [10, 100, 1000, 10000]
	methods = ['method1', 'method2', 'method3']
	to_plot = []
	legends = []
	for T in Ts:
	  for method in methods:
		to_plot.append(df.loc[(df['T'] == T) & (df['n'] > 100)][method].values)
		legends.append(method)

	fig, ax1 = plt.subplots()
	for plot in to_plot[:3]:
	  ax1.plot(N, plot, alpha=0.7)
	plt.yscale('log')
	#ax1.legend(legends[:3],  fontsize=12, bbox_to_anchor=(2,1))
	ax1.grid(True)
	#plt.xlabel('$n$', fontsize=15)
	plt.ylabel('Estiamted $\lambda/\mu$', fontsize=15)
	plt.title('Approximating $\lambda/\mu$ in the 3 Regimes, $T=10$', fontsize=15)
	ax2 = ax1.twinx()
	ax2.plot(N, As[0], 'r', linestyle="--")
	#ax2.legend(['$\\alpha$'], fontsize=12, bbox_to_anchor=(2,2))
	ax2.tick_params(axis='y', colors="red")
	plt.show()

	fig, ax1 = plt.subplots()
	for plot in to_plot[3:6]:
	  ax1.plot(N, plot, alpha=0.7)
	plt.yscale('log')
	ax1.legend(legends[3:6], bbox_to_anchor=(1.1,1), fontsize=12)
	ax1.grid(True)
	#plt.xlabel('$n$', fontsize=15)
	#plt.ylabel('Estiamted $\lambda/\mu$', fontsize=15)
	plt.title('Approximating $\lambda/\mu$ in the 3 Regimes, $T=100$', fontsize=15)
	ax2 = ax1.twinx()
	ax2.plot(N, As[1], 'r', linestyle="--")
	ax2.legend(['$\\alpha$'], fontsize=12, bbox_to_anchor=(1.29, 0.7))
	ax2.tick_params(axis='y', colors="red")
	plt.show()

	fig, ax1 = plt.subplots()
	for plot in to_plot[6:9]:
	  ax1.plot(N, plot, alpha=0.7)
	plt.yscale('log')
	#ax1.legend(legends[6:9], bbox_to_anchor=(1,1), fontsize=12)
	ax1.grid(True)
	plt.xlabel('$n$', fontsize=15)
	plt.ylabel('Estiamted $\lambda/\mu$', fontsize=15)
	plt.title('Approximating $\lambda/\mu$ in the 3 Regimes, $T=1,000$', fontsize=15)
	ax2 = ax1.twinx()
	ax2.plot(N, As[2], 'r', linestyle="--")
	#ax2.legend(['$\\alpha$'], fontsize=12)
	ax2.tick_params(axis='y', colors="red")
	plt.show()

	fig, ax1 = plt.subplots()
	for plot in to_plot[9:12]:
	  ax1.plot(N, plot, alpha=0.7)
	plt.yscale('log')
	#ax1.legend(legends[9:12], bbox_to_anchor=(1,1), fontsize=12)
	ax1.grid(True)
	plt.xlabel('$n$', fontsize=15)
	#plt.ylabel('Estiamted $\lambda/\mu$', fontsize=15)
	plt.title('Approximating $\lambda/\mu$ in the 3 Regimes, $T=10,000$', fontsize=15)
	ax2 = ax1.twinx()
	ax2.plot(N, As[3], 'r', linestyle="--")
	#ax2.legend(['$\\alpha$'], fontsize=12)
	ax2.tick_params(axis='y', colors="red")
	plt.show()

def get_resid_1(X, Y, val, weights=None):
  resid = 0
  for i in range(len(X)):
    if Y[i] != None and np.isnan(Y[i]) == False: 
      pred_EX = X[i]+(1+(val**2)/(X[i]**2))/(2*(1-val/X[i]))
      EX = X[i]+(1+(Y[i]**2)/(X[i]**2))/(2*(1-Y[i]/X[i]))
    if weights == None: resid += (EX-pred_EX)**2
    else: resid += weights[i] * ((EX-pred_EX)**2)
  return resid/len(X)

def get_resid_2(X, Y, val, alphas, weights=None):
  resid = 0
  for i in range(len(X)):
    if Y[i] != None:  
      pred_EX = val+val/(X[i]-val)*(alphas[i])
      EX = Y[i]+Y[i]/(X[i]-Y[i])*(alphas[i])
    if weights == None: resid += (EX-pred_EX)**2
    else: resid += weights[i] * ((EX-pred_EX)**2)
  return resid/len(X)

def get_resid_3(X, Y, val, weights=None):
  resid = 0
  for i in range(len(X)):
    if Y[i] != None: 
      pred_EX = val
      EX = Y[i]
    if weights == None: resid += (EX-pred_EX)**2
    else: resid += weights[i] * ((EX-pred_EX)**2)
  return resid/len(X)
  
X = df.loc[(df['T'] == 10) & (df['n'] > 100)]['n'].values
Y10_1 = df.loc[(df['T'] == 10) & (df['n'] > 100)]['method1'].values
Y100_1 = df.loc[(df['T'] == 100) & (df['n'] > 100)]['method1'].values
Y1000_1 = df.loc[(df['T'] == 1000) & (df['n'] > 100)]['method1'].values
Y10000_1 = df.loc[(df['T'] == 10000) & (df['n'] > 100)]['method1'].values

Y10_2 = df.loc[(df['T'] == 10) & (df['n'] > 100)]['method2'].values
Y100_2 = df.loc[(df['T'] == 100) & (df['n'] > 100)]['method2'].values
Y1000_2 = df.loc[(df['T'] == 1000) & (df['n'] > 100)]['method2'].values
Y10000_2 = df.loc[(df['T'] == 10000) & (df['n'] > 100)]['method2'].values

Y10_3 = df.loc[(df['T'] == 10) & (df['n'] > 100)]['method3'].values
Y100_3 = df.loc[(df['T'] == 100) & (df['n'] > 100)]['method3'].values
Y1000_3 = df.loc[(df['T'] == 1000) & (df['n'] > 100)]['method3'].values
Y10000_3 = df.loc[(df['T'] == 10000) & (df['n'] > 100)]['method3'].values

weights3 = []
for A in As:
  nomzd = normalize([[1-i for i in A]])
  weight = [i**2 for i in nomzd[0]]
  print("For method 3 the weights sum up to ", sum(weight))
  weights3.append(weight)

weights2 = []
for A in As:
  nomzd = normalize([[1/(1+np.absolute(0.5-i)) for i in A]])
  weight = [i**2 for i in nomzd[0]]
  print("For method 2 the weights sum up to ", sum(weight))
  weights2.append(weight)

weights1 = []
for A in As:
  nomzd = normalize([A])
  weight = [i**2 for i in nomzd[0]]
  print("For method 1 the weights sum up to ", sum(weight))
  weights1.append(weight)

#oserve which lambda/my value has minimal MSE  
vals = np.linspace(90, 110, 101)

resids = [get_resid_2(X, Y10000_2, val, As[3]) for val in vals]
#resids = [get_resid_3(X, Y10_3, val, weights3[0]) for val in vals]
for i in range(len(vals)):
  print(vals[i], resids[i])
  
def visualize_no_weights():
	EX = df.loc[(df['T'] == 10000) & (df['n']>100)]['EX'].values
	N = range(101, 131)
	#plt.plot(N, [N[i]+(1+(Y10000_1[i]**2)/(N[i]**2))/(2*(1-Y10000_1[i]/N[i])) for i in range(len(N))])
	plt.plot(N, EX, linewidth=3, alpha=0.8)

	val1 = 99
	method1 = [i+(1+val1**2/i**2)/(2*(1-val1/i)) for i in N]
	plt.plot(N, method1)

	alphas = As[3]
	val2 = 97.8
	method2 = [val2+val2/(i-val2)*(alphas[i-101]) for i in N]
	plt.plot(N, method2)

	val3 = 98.8
	method3 = [val3 for i in N]
	plt.plot(N, method3)

	plt.title("Theoretical $E[X]$ using best $\lambda/\mu$ - Two-Point $\mu$", fontsize=15)
	plt.xlabel('$n$', fontsize=15)
	plt.ylabel('$E[X]$', fontsize=15)
	plt.legend(['Empirical $E[X]$', 'Method 1', 'Method 2', 'Method 3'], fontsize=12)
	plt.show()
	
def visualize_weights():
	EX = df.loc[(df['T'] == 10000) & (df['n']>100)]['EX'].values
	N = range(101, 131)
	#plt.plot(N, [N[i]+(1+(Y10000_1[i]**2)/(N[i]**2))/(2*(1-Y10000_1[i]/N[i])) for i in range(len(N))])
	plt.plot(N, EX, linewidth=3, alpha=0.8)

	val1 = 20
	method1 = [i+(1+val1**2/i**2)/(2*(1-val1/i)) for i in N]
	plt.plot(N, method1)

	alphas = [0.287, 0.251, 0.22, 0.214, 0.177, 0.159, 0.157, 0.128, 0.122, 0.091, 0.092, 0.08, 0.063, 0.046, 0.038, 0.03, 0.033, 0.027, 0.027, 0.018, 0.016, 0.009, 0.009, 0.005, 0.008, 0.001, 0.002, 0.006, 0.0, 0.0]
	val2 = 97.8
	method2 = [val2+val2/(i-val2)*(alphas[i-101]) for i in N]
	plt.plot(N, method2)

	val3 = 99
	method3 = [val3 for i in N]
	plt.plot(N, method3)

	plt.title("Theoretical $E[X]$ using best $\lambda/\mu$ - Weighted", fontsize=15)
	plt.xlabel('$n$', fontsize=15)
	plt.ylabel('$E[X]$', fontsize=15)
	plt.legend(['Empirical $E[X]$', 'Method 1', 'Method 2', 'Method 3'], fontsize=12)
	plt.show()