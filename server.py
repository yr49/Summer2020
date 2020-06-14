import random

class Server:
  ID = None
  mu = None
  is_busy = False
  customer = None

  def __init__(self, ID, mu):
    self.ID = ID
    self.mu = mu

  def generateService(self):
    # This method produces a service time from a distribution
    #   currently, and exponential service time is used
    #   however, this can be changed to any chosen distribution
    return random.expovariate( 1.0 / self.mu )