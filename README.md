#  Gradient Descent for Linear Regression

This project implements the **Gradient Descent** algorithm to perform **Linear Regression**. It uses numpy to compute vectorized operations, which helps optimize the computation, making the process faster and more efficient. The purpose of this program is to train a linear regression model using the given dataset and to optimize the model parameters (weights and bias) using gradient descent.

## Overview

Linear regression is a fundamental machine learning algorithm used for predicting a continuous target variable based on one or more input features. The relationship is modeled as:

\[
y = w \cdot x + b
\]

Where:
- \(y\) is the predicted output.
- \(x\) is the input feature.
- \(w\) is the weight or coefficient of the input feature.
- \(b\) is the intercept or bias.

In this code, we perform the following:
1. **Gradient Descent**: This is an iterative optimization algorithm used to minimize the cost function (in this case, the Mean Squared Error).
2. **Linear Regression**: We aim to find the optimal values for the weight \(w\) and bias \(b\) to make the prediction \(y = w \cdot x + b\) as close as possible to the actual target values.

## Dependencies

- **NumPy**: To handle vectorized operations and compute the dot product efficiently.
- **math**: For utility functions like `math.isclose` to check for convergence based on the cost function.

### Installation

Ensure you have **NumPy** installed. You can install it using:

```bash
pip install numpy
```

## Code Description

### 1. Data Setup
The input data (`x`) and target data (`y`) are represented as numpy arrays:

```python
x = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
y = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])
```

In this example, `x` represents the features (input data), and `y` represents the target values (output data).

### 2. Gradient Descent Function (`G`)
The function `G(X, y_actual, lr, n_iters)` is defined to perform linear regression using gradient descent:

```python
def G(X, y_actual, lr, n_iters):
  w, b = 0, 0
  n = len(X)
  prev_cost = 0

  for i in range(n_iters):
    y_pred = X * w + b
    cost_func = (1/n) * np.sum((y_pred - y_actual)**2)
    dw = (1/n) * np.dot(X.T, (y_pred - y_actual))
    db = (1/n) * np.sum(y_pred - y_actual)

    w -= lr * dw
    b -= lr * db

    if math.isclose(cost_func, prev_cost, rel_tol=1e-09, abs_tol=0.0):
      break

    prev_cost = cost_func
    print(f'w({w}), b({b}), i({i}), cost({cost_func})')

  return w, b
```

#### Parameters:
- **X**: The input features (independent variables) of the dataset.
- **y_actual**: The actual target values.
- **lr**: The learning rate, which controls the step size during optimization.
- **n_iters**: The maximum number of iterations to perform.

#### Process:
1. **Initial Parameters**: The initial values for the weight `w` and the bias `b` are set to zero.
2. **Prediction**: The predicted values are calculated as:
   \[
   y_{pred} = w \cdot x + b
   \]
3. **Cost Function**: The cost function (Mean Squared Error) is calculated:
   \[
   \text{cost} = \frac{1}{n} \sum_{i=1}^n (y_{pred} - y_{actual})^2
   \]
4. **Gradient Calculation**: The gradients of the cost function with respect to `w` and `b` are computed:
   \[
   dw = \frac{1}{n} \cdot \sum (y_{pred} - y_{actual}) \cdot x
   \]
   \[
   db = \frac{1}{n} \sum (y_{pred} - y_{actual})
   \]
5. **Parameter Update**: The weight `w` and bias `b` are updated using the learning rate:
   \[
   w = w - \text{lr} \cdot dw
   \]
   \[
   b = b - \text{lr} \cdot db
   \]
6. **Convergence Check**: The algorithm checks if the cost function has converged. If the cost function value does not change significantly between iterations, the algorithm stops early.

### 3. Main Execution
In the main block, the function `G` is called to optimize the parameters \(w\) and \(b\):

```python
if __name__ == "__main__":
    w, b = G(x, y, 0.0001, 1000)
    print(f'Coef {w:.4f} Intercept {b:.4f}')
```

- The **learning rate** is set to `0.0001`.
- The **maximum number of iterations** is set to `1000`.
- The optimized values of the weight `w` and bias `b` are printed to the console after convergence.

### 4. Output

During the execution, the algorithm prints the values of `w`, `b`, the iteration number, and the cost at each iteration:

```
w(0.1234), b(0.2345), i(0), cost(123.4567)
w(0.1245), b(0.2356), i(1), cost(122.3456)
...
```

At the end, the final values of `w` and `b` are printed:

```
Coef 0.2345 Intercept 45.6789
```

These values represent the optimized parameters for the linear regression model.

## Hyperparameters

- **Learning Rate (lr)**: The learning rate controls the step size for updating the model parameters. A larger value may lead to overshooting the optimal solution, while a smaller value can lead to slow convergence.
- **Number of Iterations (n_iters)**: The maximum number of iterations for the gradient descent. If the cost function converges before reaching the maximum iterations, the algorithm stops early.
- **Tolerance for Convergence**: The algorithm stops early if the cost function value does not change significantly between iterations. The threshold for this change is set with a relative tolerance of `1e-9`.

## Example Usage

To run the algorithm, you can execute the Python script:

```bash
python gradient_descent.py
```

The script will output the optimized parameters for the linear regression model and display the progress during the optimization.

## Limitations

- This implementation is based on a simple linear regression model (single feature). It can be extended to handle multiple features by adjusting the code to use matrix operations.
- The code does not include feature scaling. It assumes that the features are already normalized or scaled appropriately.
- The learning rate is fixed in this implementation, but it can be dynamically adjusted based on certain conditions (e.g., using a learning rate scheduler).

## Conclusion

This project demonstrates the implementation of **gradient descent** for **linear regression** using Python. It optimizes the weight and bias parameters of the model to minimize the cost function and make the model's predictions as accurate as possible. You can modify this implementation to suit more complex models or datasets with multiple features.
