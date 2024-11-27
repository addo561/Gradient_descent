
import numpy as np 
import math

x = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
y = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])

def G(X,y_actual,lr,n_iters):
  w,b = 0,0
  n = len(X)
  prev_cost = 0

  for i in range(n_iters):
    y_pred = X * w + b
    cost_func = (1/ n) * np.sum((y_pred - y_actual)**2) 
    dw = (1/n) * np.dot(X.T,(y_pred - y_actual))
    db = (1/n) * np.sum(y_pred - y_actual)

    w -= lr * dw
    b -= lr * db

    if math.isclose(cost_func,prev_cost,rel_tol=1e-09,abs_tol=0.0):
      break

    prev_cost = cost_func
    print(f'w({w}), b({b}), i({i}), cost({cost_func})')
      
  return w,b 

if __name__ == "__main__":
    w, b = G(x, y,0.0001,1000)
    print(f'Coef {w:.4f} Intercept {b:.4f}')

 

    