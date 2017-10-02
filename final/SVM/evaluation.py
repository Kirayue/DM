import numpy as np
from pprint import pprint

def readpred():
    with open('test', 'r') as f:
        rawdata = f.read()
        rawdata = rawdata.split('\n')
        del rawdata[-1]
    return np.array(rawdata, dtype = np.float32)
def readans():
    with open('test_ans', 'r') as f:
        rawdata = f.read()
        rawdata = rawdata.split('\n')
        del rawdata[-1]
    return np.array(rawdata, dtype = np.float32)

def MSE(pred, ans):
    return np.sum((pred - ans) * (pred - ans)) / len(pred)
    
def MAE(pred, ans):
    return np.sum(np.abs(pred - ans)) / len(pred)
if __name__ == '__main__':
    pred = readpred()
    ans = readans()
    pprint(type(pred)) 
    pprint(type(ans))
    mse = MSE(pred, ans)
    mae = MAE(pred, ans)
    print("MSE: {}, MAE: {}".format(mse, mae))
