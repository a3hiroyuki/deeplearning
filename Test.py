'''
Created on 2017/09/04

@author: hiroy
'''

import numpy as np
if __name__ == '__main__':
    
    a = np.zeros((4,2))
    
    b = np.arange(4)
    
    a[b, [0,1,1,1]] = 1
    
    
    print (a)
    
    #rint (a[b, [1,1,1,1]])
    