import numpy as np

def Xavier_normal(fan_in, fan_out = 1):  
    return np.random.normal(0, np.sqrt(2/(fan_in + fan_out)), fan_in)

def Xavier_uniform(fan_in, fan_out = 1):
    param = np.sqrt(6/(fan_in + fan_out))
    return np.random.uniform(-param, param, fan_in)

    