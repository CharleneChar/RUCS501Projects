## RU CS501 Projects :clipboard:
Hello there!\
This repository holds my programming projects done during the period of Rutgers University CS501 course - Math for Data Science.


### :pushpin: Project Objectives and Execution:
* Homework 1 

  It approximates a square root of a D x D square matrix $B$ obtained from $\tilde{A}^{T}\tilde{A}$ where $\tilde{A}$ is a k x D matrix, i.e., approximating $A$ where $B = A^TA$ (which might or might not be $\tilde{A}$), by minimizing $L(A) = \displaystyle \sum_{i=1}^{D} \sum_{j=1}^{D} [B - A^TA]^2_{i, j}$ with gradient descent implementation.

  To see the process and result,
    1. Clone this repository (PS you may follow the command below) \
  `git clone https://github.com/CharleneChar/RUCS501Projects.git`
    2. Execute the main.py under Homework_1 directory (PS you may follow the command below) \
  `python3 Homework_1/main.py`

* Homework 2

  It achieves two things. 
  
  The first one approximates a solution to a quadratic function expressed in matrix form, i.e., approximating $\underline x$ for minimizing $F(\underline x) = \frac{1}{2} \underline x^{T} Q \underline x - \underline x^{T} \underline c$ with different descent methods including gradient descent (where both constant and optimal hyperparameters are attempted), gradient descent with momentum (where both constant and optimal hyperparameters are attempted), and conjugate gradient descent. 
  
  The second one solves a constrained problem of maximizing a polynomial function such that an inequality is satisfied, i.e., finding $x_1, x_2, ..., x_{10} \in \mathbb{Z}$ to maximize $104x_1 + 128x_2 + 135x_3 + 139x_4 + 150x_5 + 153x_6 + 162x_7 + 168x_8 + 195x_9 + 198x_{10}$ such that $9x_1^{2} + 8x_2^{2} + 7x_3^{2} + 7x_4^{2} + 6x_5^{2} + 6x_6^{2} + 5x_7^{2} + 2x_8^{2} + x_9^{2} + x_{10}^{2} \leq 68644$ with branch and bound algorithm.

  To see the process and result,
    1. Clone this repository (PS you may follow the command below) \
  `git clone https://github.com/CharleneChar/RUCS501Projects.git`
    2. Execute the main.py under Homework_2 directory (PS you may follow the command below) \
  `python3 Homework_2/main.py`

* Homework 3

  It achieves two things. 
  
  The first one calculates probability of whether a party will win in an election (where there might or might not have gerrymandering). 
  
  The second one calculates probability $P(X > 1|Y > 1)$ and $P(X > 1)$ from joint distribution, given the fact that with probability 1/2, $X$ and $Y$ are i.i.d. normal random variables, with mean $μ = 0$ and variance $σ^2 = 1$; with probability 1/2, $X$ and $Y$ are i.i.d. normal random variables with mean $μ = 0$ and variance $σ^2 = 2$.

  To see the results,
    1. Clone this repository (PS you may follow the command below) \
  `git clone https://github.com/CharleneChar/RUCS501Projects.git`
    2. Execute the main.py under Homework_3 directory (PS you may follow the command below) \
  `python3 Homework_3/main.py`


### :pushpin: Project Source Code:
* [Homework 1]( https://github.com/CharleneChar/RUCS501Projects/blob/main/Homework_1/main.py)
* [Homework 2]( https://github.com/CharleneChar/RUCS501Projects/blob/main/Homework_2/main.py)
* [Homework 3]( https://github.com/CharleneChar/RUCS501Projects/blob/main/Homework_3/main.py)

