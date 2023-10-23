import pyvisa
import numpy as np
import matplotlib.pyplot as plt

rm = pyvisa.ResourceManager()
print(rm.list_resources())