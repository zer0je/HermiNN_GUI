import anvil.server
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from IPython.display import clear_output
import time
from sklearn.metrics import mean_squared_error, r2_score
from scipy.linalg import solve
from scipy.interpolate import interp1d

@anvil.server.callable
def hello(x):
  print('Hello')