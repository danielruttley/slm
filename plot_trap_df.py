# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:22:20 2021

@author: mfqq76
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'Z:\Tweezer\Code\Python 3.7\slm\images\2021\November\23\Measure 3\trap_df.csv')
plt.scatter(df['img_x'],df['img_y'])
plt.show()