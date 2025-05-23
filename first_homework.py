import os
os.system('pip install numpy')
os.system('pip install scipy')
os.system('pip install matplotlib')
import numpy as np
from scipy import interpolate
from scipy.linalg import lu, solve_triangular
from scipy.optimize import fsolve, minimize, golden, minimize_scalar,linprog
from scipy.interpolate import CubicSpline, RectBivariateSpline
import matplotlib.pyplot as plt

# === Exercise 2.1 ===

A = np.array([
    [1, 5, 2, 3],
    [1, 6, 8, 6],
    [1, 6, 11, 2],
    [1, 7, 17, 4]
])
b = np.array([1, 2, 1, 1])
P, L, U = lu(A)
y = np.linalg.solve(L, P @ b)
x = np.linalg.solve(U, y)
print("Exercise 2.1: LU Decomposition")
print(f'product of L and U{L@U}')
print("Solution x =", x)
print(f'Product of A and b{A@b}')
# === Exercise 2.2===
import random
def system(vars):
    c1, c2, lam = vars
    return [
        1 / c1**2 - lam,
        1 / c2**2 - lam,
        c1 + c2 - 1
    ]
sol = fsolve(system, [0.4, 0.6, 1])
c1, c2, lam = sol
print(f"c1 = {c1:.4f}, c2 = {c2:.4f}, λ = {lam:.4f}")
def U(c): 
    c1, c2 = c
    if c1 <= 0 or c2 <= 0: #check if the initial guesses are positive
        return np.inf   #if not, return infinity  
    return 1 / c1 + 1 / c2
constraint = { 
    'type': 'eq',
    'fun': lambda c: c[0] + c[1] - 1
}
result = minimize(U, [0.4, 0.6], method='SLSQP', constraints=constraint)
c1_opt, c2_opt = result.x
print("Solution using minimization :")
print(f"c1 = {c1_opt:.4f}")
print(f"c2 = {c2_opt:.4f}")
print(f"Utility at the equilibrium = {U([c1_opt, c2_opt]):.4f}")

#2.3
def F(x):return x*np.cos(x**2)
def golden_search_minimize (F, lower_bound, upper_bound):return golden(F,brack=(lower_bound,upper_bound)) 
interval = [0,5]
subinterval_num = 6
subinterval_bounds = np.linspace(interval[0],interval[1],subinterval_num) 
subintervals = [(subinterval_bounds[i],subinterval_bounds[i+1]) for i in range(len(subinterval_bounds)-1)] 
minima = [golden_search_minimize(F,subintervals[i][0],subintervals[i][1]) for i in range(len(subintervals))] 
global_minimum_location = np.argmin(minima) 
for i in range(len(subintervals)):
    print(f'The local minimum on [{subintervals[i][0]},{subintervals[i][1]} is {minima[i]:.4f}]\n') 
print(f'\nThe global minimum is {minima[global_minimum_location]}') 

#2.4
gamma, beta = 0.5, 1
def neg_U(c, r, w):
    c2, c3 = c 
    c1 = (w + w / (1 + r)) - c2 / (1 + r) - c3 / ((1 + r)**2) 
    if c1 <= 0 or c2 <= 0 or c3 <= 0: 
        return np.inf  
    u1 = c1**(1 - 1/gamma) / (1 - 1/gamma) 
    u2 = beta * c2**(1 - 1/gamma) / (1 - 1/gamma)
    u3 = beta**2 * c3**(1 - 1/gamma) / (1 - 1/gamma)
    return -(u1 + u2 + u3) 
def optimize(r, w): 
    result = minimize(neg_U, x0=[0.5, 0.5], args=(r, w), method='SLSQP') 
    c2_opt, c3_opt = result.x 
    c1_opt =  w + w / (1 + r)  - c2_opt / (1 + r) - c3_opt / ((1 + r)**2) 
    U = neg_U((c2_opt, c3_opt), r, w) * (-1)
    return U
r, w = 0, 1
rhs = w + w / (1 + r) 
initial_guess = [0.3, 0.3] 
result = minimize(neg_U, initial_guess, args=(r, w), method='SLSQP') 
optimal_c2, optimal_c3 = result.x 
optimal_c1 = rhs - optimal_c2 / (1 + r) - optimal_c3 / ((1 + r)**2) 
optimal_utility = neg_U([optimal_c2, optimal_c3],r,w) 
print(f"Optimal consumption:\noptimal c1 = {optimal_c1:.4f}, optimal c2 = {optimal_c2:.4f}, optimal c3 = {optimal_c3:.4f}\nMaximum utility = {optimal_utility:.4f}") 
r_vals = np.linspace(0.001, 0.2, 50) 
utilities_r = [optimize(r, w=1.0) for r in r_vals] 
plt.figure(figsize=(8, 5)) 
plt.plot(r_vals, utilities_r) 
plt.xlabel('(r)') 
plt.ylabel('U*') 
plt.title('U(r)') 
plt.grid(True) 
plt.show()
w_vals = np.linspace(0.5, 2.0, 50) 
utilities_w = [optimize(r=0.05, w=w) for w in w_vals] 
plt.figure(figsize=(8, 5)) 
plt.plot(w_vals, utilities_w) 
plt.xlabel('w') 
plt.ylabel('U*') 
plt.title('U(w)') 
plt.grid(True) 
plt.show()

#2.7

tau = np.array([37, 42, 45]) 
T = np.array([198.875, 199.5, 196.875]) 
coefficients = np.polyfit(tau,T,deg=tau.size-1) 
polynomial = np.poly1d(coefficients) 
print(f'polynomial: {polynomial[2]}x^{tau.size-1} + {polynomial[1]}x^{tau.size-2} + {polynomial[0]}')
tau_range = np.linspace(35, 45, 200) 
T_vals = polynomial(tau_range) 
result = minimize_scalar(lambda t: -polynomial(t), bounds=(35, 45), method='bounded') 
tau_opt = result.x 
T_opt = polynomial(tau_opt) 
print(f"τ* = {tau_opt:.4f}")
print(f"T(τ*) = {T_opt:.4f}")
plt.plot(tau_range, T_vals) 
plt.scatter(tau, T, color='red') 
plt.xlabel("τ") 
plt.ylabel("T(τ)") 
plt.grid(True) 
plt.show()

#2.9
alpha = 1.0 
eta = 1.5 
m = 3 
N = 10 
NP = 1000 
P_grid = np.linspace(0.1, 3.0, N + 1) 
def D(P):
    return P ** (-eta) 
def md(P):
    return -eta * P ** (-eta - 1)
def mc(q):
    return alpha * np.sqrt(q) + q**2
def firm_q(P):
    def F(q):
        MB = P + q * (1 / md(P)) * (1 / m) 
        MC = mc(q) 
        return MB - MC
    q_guess = D(P) / m 
    q_sol, = fsolve(F, q_guess) 
    return max(q_sol, 0)  
def ed(P):
    if P <= 0:
        return 1e6  
    return D(P) - m * spline_q(P)
q_vals = np.array([firm_q(P) for P in P_grid])
spline_q = CubicSpline(P_grid, q_vals)
P_fine = np.linspace(0.1, 3.0, NP) 
q_interp = spline_q(P_fine) 
P_eq_guess = 1.0
P_eq, = fsolve(ed, P_eq_guess)
Q_eq = D(P_eq)
plt.figure(figsize=(10, 6)) 
plt.plot(P_fine, D(P_fine)) 
plt.plot(P_fine, m * q_interp) 
plt.xlabel("P") 
plt.ylabel("Q") 
plt.show()
print(f"Equilibrium price: {P_eq:.4f}, Quantity: {Q_eq:.4f}")
print("Increasing m leads to more competition, pushing price down and quantity up.\nIncreasing α makes costs higher, reducing supply and increasing price.\nIncreasing η makes demand more elastic: flattens demand, leading to more responsive price changes.") 


#2.10
P_r, P_a = np.linspace(0.5, 12.5, 4), np.linspace(0.5, 12.5, 4)
G = np.array(
    [
    [11.5, 70.9, 98.3, 93.7],
    [31.1, 82.5, 101.9, 89.3],
    [18.7, 62.1, 73.5, 52.9],
    [-25.7, 9.7, 13.1, -15.5]
]
)
spline_G = interpolate.RectBivariateSpline(P_r, P_a, G)
Nplot = 100 
P_r_fine, P_a_fine = np.linspace(0.5, 12.5, Nplot), np.linspace(0.5, 12.5, Nplot) 
profit_matrix = spline_G(P_r_fine, P_a_fine) 
max_idx = np.unravel_index(np.argmax(profit_matrix), profit_matrix.shape) 
P_r_optimal, P_a_optimal, profit_optimal = P_r_fine[max_idx[0]], P_a_fine[max_idx[1]], profit_matrix[max_idx] 
print(f"(a) Optimal prices from interpolation: pR* = {P_r_optimal:.2f}, pA* = {P_a_optimal:.2f}\nMaximum approximated profit: G(pR*, pA*) = {profit_optimal:.2f}")
def true_profit(prices):
    P_r, P_a = prices
    D_r = 10 - P_r 
    D_a = 20 - P_a - 0.5 * P_r  
    profit = (P_r - 0.1) * D_r + (P_a - 0.1) * D_a 
    return profit*(-1)  
res = minimize(true_profit, x0=[random.uniform(0, 1), random.uniform(0, 1)], bounds=[(0.5, 12.5), (0.5, 12.5)])
P_r_opt, P_a_opt = res.x
profit_opt_true = -res.fun 
print(f"(b) Optimal prices from true profit: pR* = {P_r_opt:.2f}, pA* = {P_a_opt:.2f}\nMaximum true profit: G(pR*, pA*) = {profit_opt_true:.2f}")
for Nplot in [100, 1000, 10000]:
    P_r_fine, P_a_fine = np.linspace(0.5, 12.5, Nplot), np.linspace(0.5, 12.5, Nplot)
    profit_matrix = spline_G(P_r_fine, P_a_fine)
    max_idx = np.unravel_index(np.argmax(profit_matrix), profit_matrix.shape)
    P_r_optimal, P_a_optimal, profit_optimal = P_r_fine[max_idx[0]], P_a_fine[max_idx[1]], profit_matrix[max_idx]
    error = abs(profit_optimal - profit_opt_true)
    print(f"Nplot = {Nplot}: Approx Profit = {profit_optimal:.2f}, Error = {error:.4f}")

#2.11
costs = np.array([
    10, 70, 100, 80, 0,  
    130, 90, 120, 110, 0, 
    50, 30, 80, 10, 0     
])
S = [11, 13, 10]
D = [5, 7, 13, 6, 3]
source_num = 3
destination_num = 5
n_variables = source_num * destination_num 
S_constraints   = np.zeros((source_num, n_variables)) 
for i in range(source_num):
    for j in range(destination_num):
        S_constraints[i, i * destination_num + j] = 1 
D_constraints = np.zeros((destination_num, n_variables)) 
for j in range(destination_num):
    for i in range(source_num):
        D_constraints[j, i * destination_num + j] = 1 
A_eq = np.vstack([S_constraints, D_constraints]) 
b_eq = np.array(S + D) 
bounds = [(0, None)] * n_variables 
res = linprog(c=costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs') 
X = res.x.reshape((source_num, destination_num)) 
print("Optimal transport plan:")
for i in range(source_num):
    for j in range(destination_num):
        if X[i, j] > 0:
            print(f"A{i+1} -> B{j+1}: {X[i, j]:.2f} tons") 
print(f"\nMinimum total cost: {res.fun:.2f}") 

