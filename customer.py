class Customer:
  ID = None
  t_arr = None
  t_serv = None
  t_dep = None

  def __init__(self, ID, t_arr):
    self.ID = ID
    self.t_arr = t_arr