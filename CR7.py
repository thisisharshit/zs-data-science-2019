# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:16:25 2019

@author: harshit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv")
del data['Unnamed: 0']
