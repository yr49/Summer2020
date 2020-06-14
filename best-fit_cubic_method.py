import matplotlib.pyplot as plt
from run_experiment import run_experiment

x = range(101, 126) # This is the number of servers in the system

T = 10

c = 5
h = 15

cost = []

for N in x:
  lm, mu, EQ = run_experiment(N, T)

  cost.append(c*N + h*EQ)
  
x1 = range(101, 126)
x2 = range(101, 126, 2)
x3 = range(101, 126, 3)
x4 = range(101, 126, 4)
x5 = range(101, 126, 5)
xs = [x1, x2, x3, x4, x5]

plt.scatter(x1, cost)
for x in xs:
  y = [cost[101-i] for i in x]
  mymodel = np.poly1d(np.polyfit(x, y, 3))
  myline = np.linspace(101, 125, 100)
  plt.plot(myline, mymodel(myline))
plt.axvline(112, color='k', linestyle='dotted')
plt.legend(["|X|=25", "|X|=13", "|X|=9", "|X|=7", "|X|=5"])
plt.grid(True)
plt.xlabel('$n$', fontsize=15)
plt.ylabel('Cost', fontsize=15)
plt.title("Statistical Method: $T$=10", fontsize=15)
plt.show()