
import numpy as np 
import math

x = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
y = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])

def gradient_descent(x, y):
    w, b = 0, 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.0001
    cost_previous = 0

    for i in range(iterations):
        y_predict = w * x + b
        cost_function = (1 / n) * sum((value ** 2) for value in (y - y_predict))
        wd = -(2 / n) * sum(x * (y - y_predict))
        bd = -(2 / n) * sum(y - y_predict)
        m -= learning_rate * wd
        b -= learning_rate * bd

        if math.isclose(cost_function, cost_previous, rel_tol=1e-09, abs_tol=0.0):
            break
        

        cost_previous = cost_function
        print(f'm({w}), b({b}), i({i}), cost({cost_function})')

    return m, b

if __name__ == "__main__":
    w, b = gradient_descent(x, y)
    print(f'Coef {w} Intercept {b}')
    #
    
#Stochastic GD
    