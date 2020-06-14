from customer import Customer
from server import Server
	
class System:
  time = 0 # This variable keeps track of the time
  lm = 0 # This variable serves as lambda, the mean arrival rate
  servers = []
  queue = [] # This list contains all customers awaiting service
  fel = [] # This list keeps track of future events (arrivals and departures)
  verbose = False
  numInQueue = 0
  numInService = 0
  numInSystem = 0
  custRepo = [] # This list is a repository of all customers over time
  numArrivals = 0
  servCompletns = 0

  def __init__(self, servers, lm, verbose=False):
    self.lm = lm
    self.servers = servers
    self.queue = []
    self.fel = []
    self.verbose = verbose
    self.custRepo = []
    self.numInQueue = 0
    self.numInService = 0
    self.numInSystem = 0
    self.numArrivals = 0
    self.servCompletns = 0

  def get_num_in_queue(self):
    # This method returns the number of customers curently awaiting service
    return len(self.queue)
  
  def get_num_in_service(self):
    # This method returns the number of customers currently in service
    res = 0
    for server in self.servers:
      if server.is_busy: res += 1
    return res

  def get_num_in_system(self):
    # This method returns the number of customers currently in the system
    return self.get_num_in_queue() + self.get_num_in_service()

  def updateStats(self, eventTime):
    # This method updates three statistics over time:
    #   total queue length 
    #   total number of customers in service 
    #   total number of customers in the system
    dt = eventTime - self.time
    self.numInQueue += self.get_num_in_queue() * dt
    self.numInService += self.get_num_in_service() * dt
    self.numInSystem += self.get_num_in_system() * dt

  def find_free_server(self):
    # This method returns any free server, or None if all servers are busy
    for server in self.servers:
      if server.is_busy == False:
        return server
    return None

  def generateInterarrival(self):
    # This method returns an inter-arrival time:
    #   Currently, an expolnential interarrival time is being used
    #   This can be modified to utilize any other distribution
    return random.expovariate( 1.0 / self.lm )

  def addEvent(self, eventType, eventTime):
    # This method adds an event to the future event list
    self.fel.append((eventType, eventTime))

  def deleteEvent(self, index):
    # This method removes an event from the future event list
    self.fel.pop(index)

  def findNextEvent(self):
    # This method find the next event from the future event list
    earliestTime = 1e30
    earliestIndex = -1
    for i in range( len( self.fel ) ):
      event = self.fel[ i ]
      eventTime = event[ 1 ]
      if ( eventTime < earliestTime ):
        earliestTime = eventTime
        earliestIndex = i
    return earliestIndex

  def handleArrival(self, customer, eventTime):
    # When an arrival occurs, this method routes the customer accordingly
    self.updateStats(eventTime)
    self.numArrivals += 1
    if self.verbose:
      print()
      print("customer ", customer.ID, " arrives at time ", eventTime)
    self.custRepo.append(customer)
    server = self.find_free_server()
    if server == None: # If no server is free
      self.queue.append(customer) # The customer is sent to the back of the queue
      if self.verbose:print("no servers available, queue length is ", len(self.queue))
    else: # If a free server exists
      if self.verbose:print("server ", server.ID, " is free")
      st = server.generateService() # The next customer's service time is generated
      self.addEvent("d"+str(server.ID), st + eventTime) # Service completion event is scheduled
      self.serviceStart(server, customer, eventTime) 
      if self.verbose:print("service completion scheduled for ", st+eventTime)
    next_arrival = eventTime + self.generateInterarrival() 
    self.addEvent("a", next_arrival) # The following arrival is scheduled
    self.time = eventTime
    if self.verbose:print("next arrival at time ", next_arrival)

  def handleServiceComp(self, server, eventTime):
    # When a service completion occurs, this method routes awaiting customers accordingly
    self.updateStats(eventTime)
    self.servCompletns += 1
    if self.verbose:
      print()
      print("server ", server.ID, "is finished with customer ", server.customer.ID, "at time ", eventTime)
    server.is_busy = False
    server.customer.t_dep = eventTime
    server.customer = None
    if len(self.queue) > 0: # If there are customers waiting in the queue
      if self.verbose:print("queue is ", len(self.queue), "customers long")
      st = server.generateService() # The next customer's service time is generated
      self.addEvent("d"+str(server.ID), st + eventTime) # Service completion event is scheduled 
      
      self.serviceStart(server, self.queue.pop(0), eventTime) # An awaiting customer begins service, 
      #    currently the first customer in line is chosen,
      #    but, this can be changed to take a customer from anywhere in the queue
      
      if self.verbose:print("service completion scheduled for ", st+eventTime)
    self.time = eventTime

  def serviceStart(self, server, customer, t_serv):
    # This method assigns `customer` to `server` for service beginning at time `t_serv`
    if self.verbose:
      print()
      print("server ", server.ID, "takes into service customer ", customer.ID, " at time ", t_serv)
    server.is_busy = True
    server.customer = customer
    customer.t_serv = t_serv # Update the customers time of beginning service

  def getServer(self, ID):
    # Given an ID, this method returns the respective server
    for server in self.servers:
      if server.ID == ID:
        return server
    return None