from server import Server
from system import System

def run_experiment(n, T, dictn=None):

  servers = [Server(i, 1.0) for i in range(n)] # This array initializes `N` servers with the same mean service time,
  #  but it can be modified to hold any combination of servers

  system = System(servers, 0.01) # a mean inter-arrival time of 0.1 is used

  ia = system.generateInterarrival() # The first customer's arrival time is generated
  system.addEvent( "a" , ia )

  counter = 0
  next_cutoff = 10
  while system.time < T:
    eventIndex = system.findNextEvent()
    event = system.fel[ eventIndex ]
    eventType = event[ 0 ]
    eventTime = event[ 1 ]

    if eventType == "a": # An event marked by "a" is an arrival
    
      system.handleArrival( Customer(counter, eventTime), eventTime )
      counter += 1

    elif eventType[0] == "d":# An event marked by "dX" is a service completion, where X is the ID of the server completing service
      
      server = system.getServer(int(eventType[1:]))
      if server != None:
        system.handleServiceComp(server, eventTime)
      else: print( "invalid server ID")
    
    else:print( "invalid event type" )
    system.deleteEvent( eventIndex )

    if system.time > next_cutoff and dictn != None:
      EQ = system.numInQueue/system.time
      dictn[(n, next_cutoff)].append(EQ)
      #print(" At time point ", system.time, ", EQ is ", EQ)
      next_cutoff = next_cutoff * 10

  lm = system.numArrivals/T
  totServiceTime = 0
  totServed = 0
  for cust in system.custRepo:
    if cust.t_dep != None:
      totServiceTime += cust.t_dep - cust.t_serv
      totServed += 1
  mu = totServed/totServiceTime
  #print("empirical lm: ", lm)
  #print("empirical mu: ", mu)

  EQ = system.numInQueue/T
  del system
  
  return lm, mu, EQ