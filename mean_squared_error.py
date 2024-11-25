import numpy as np

y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])

def MSE(y_predicted,y_true):
    total_error  = 0
    for yp,yt in zip(y_predicted,y_true):
        total_error += (yp-yt)**2
    print(f'total error: {total_error}')
    MSE = total_error/len(y_predicted)
    print(f'MSE: {MSE}')
    
    return MSE
        
MSE(y_predicted,y_true)       
