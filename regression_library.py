import sys
sys.path.append('..')
import numpy as np
import pandas as pd 


def low_high_param(mid, step, param=3):
    if param == 3:
        low = mid - step
        high = mid + step + 1
    else:
        low = mid - (step * 2)
        high = mid + (step * 2) + 1
    return low, high





    



