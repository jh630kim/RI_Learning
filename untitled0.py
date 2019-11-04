# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:09:21 2019

@author: R570
"""

import numpy as np

Q = np.zeros([64])
for i in range(64):
    Q[i:i+1] = i*0.1

Q = Q.reshape([16,4])
print(Q)

Q1 = Q.reshape(64)

print(" 1L   1D   1R   1U  |",
      " 2L   2D   2R   2U  |",
      " 3L   3D   3R   3U  |",
      " 4L   4D   4R   4U  |")
for i in range(4):
    for j in range(16):
        print('{:.2f} '.format(Q.reshape(64)[i*16+j]),end='')
        if (j+1) % 4 == 0:
            print('| ', end='')            
    print('')
        
