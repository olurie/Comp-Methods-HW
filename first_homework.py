!!pip install numpy scipy matplotlib
import numpy as np
from scipy import interpolate
from scipy.linalg import lu, solve_triangular
from scipy.optimize import fsolve, minimize, golden, minimize_scalar,linprog
from scipy.interpolate import CubicSpline, RectBivariateSpline
import matplotlib.pyplot as plt

# === Exercise 2.1: LU Decomposition ===

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
# === Exercise 2.2: Nonlinear system (Lagrangian method) ===

def system(vars):
    c1, c2, lam = vars
    return [
        1 / c1**2 - lam,
        1 / c2**2 - lam,
        c1 + c2 - 1
    ]

sol = fsolve(system, [0.3, 0.7, 1.0]) # [0.3, 0.7, 1.0] is the initial guess
c1, c2, lam = sol
print("\nExercise 2.2: Optimal consumption")
print(f"c1 = {c1:.4f}, c2 = {c2:.4f}, λ = {lam:.4f}")

def neg_utility(c): 
    c1, c2 = c  
    return 1 / c1 + 1 / c2  # Negative of (-1/c1 - 1/c2)

# Constraint: c1 + c2 = 1
constraint = { 
    'type': 'eq',
    'fun': lambda c: c[0] + c[1] - 1
}
result = minimize(neg_utility, [0.3, 0.7], method='SLSQP', constraints=constraint)

# Extract results
c1_opt, c2_opt = result.x

print("Solution using minimization :")
print(f"c1 = {c1_opt:.4f}")
print(f"c2 = {c2_opt:.4f}")
print(f"Utility at the equilibrium = {-neg_utility([c1_opt, c2_opt]):.4f}")

# === Exercise 2.3: Interpolation ===

#2.3

def function (x):
    return x*np.cos(x**2)

def golden_search_minimize (function, lower_bound, upper_bound):

    return golden(function,brack=(lower_bound,upper_bound)) 

interval = [0,5]
subinterval_num = 6
subinterval_bounds = np.linspace(interval[0],interval[1],subinterval_num) 
subintervals = [(subinterval_bounds[i],subinterval_bounds[i+1]) for i in range(len(subinterval_bounds)-1)] 

minima = [golden_search_minimize(function,subintervals[i][0],subintervals[i][1]) for i in range(len(subintervals))] 
global_minimum_location = np.argmin(minima) 

for i in range(len(subintervals)):
    print(f'The local minimum on [{subintervals[i][0]},{subintervals[i][1]} is {minima[i]:.4f}]\n') 

print(f'\nThe global minimum is {minima[global_minimum_location]}') 

#2.4

gamma = 0.5
beta = 1

def neg_U(c, r, w):
    c2, c3 = c 
    rhs = w + w / (1 + r) 
    c1 = rhs - c2 / (1 + r) - c3 / ((1 + r)**2) 
    if c1 <= 0 or c2 <= 0 or c3 <= 0: 
        return np.inf 
    u1 = c1**(1 - 1/gamma) / (1 - 1/gamma) 
    u2 = beta * c2**(1 - 1/gamma) / (1 - 1/gamma)
    u3 = beta**2 * c3**(1 - 1/gamma) / (1 - 1/gamma)
    return -(u1 + u2 + u3) 

def optimize_given_r_w(r, w): 
    result = minimize(neg_U, x0=[0.5, 0.5], args=(r, w), method='SLSQP') 
    c2_opt, c3_opt = result.x 
    rhs = w + w / (1 + r) 
    c1_opt = rhs - c2_opt / (1 + r) - c3_opt / ((1 + r)**2) 
    utility = -neg_U((c2_opt, c3_opt), r, w)  
    return utility


# --- Single optimization at baseline ---
r = 0
w = 1
rhs = w + w / (1 + r) 
result = minimize(neg_U, x0=[0.3, 0.3], args=(r, w), method='SLSQP') 

c2_opt, c3_opt = result.x 
c1_opt = rhs - c2_opt / (1 + r) - c3_opt / ((1 + r)**2) 
utility = neg_U((c2_opt, c3_opt), r, w) 

print(f"Optimal consumption:\noptimal c1 = {c1_opt:.4f}, optimal c2 = {c2_opt:.4f}, optimal c3 = {c3_opt:.4f}\nMaximum utility = {-utility:.4f}") 

# --- Plot 1: Utility vs. Interest Rate ---
r_vals = np.linspace(0.001, 0.2, 50) 
utilities_r = [optimize_given_r_w(r, w=1.0) for r in r_vals] 

plt.figure(figsize=(8, 5)) 
plt.plot(r_vals, utilities_r, label='Utility vs. Interest Rate', color='blue') 
plt.xlabel('Interest Rate (r)') 
plt.ylabel('Max Utility') 
plt.title('Utility as a Function of Interest Rate') 
plt.grid(True) 
plt.legend() 
plt.tight_layout()
plt.show()

# --- Plot 2: Utility vs. Wage Rate ---
w_vals = np.linspace(0.5, 2.0, 50) 
utilities_w = [optimize_given_r_w(r=0.05, w=w) for w in w_vals] 

plt.figure(figsize=(8, 5)) 
plt.plot(w_vals, utilities_w, label='Utility vs. Wage Rate', color='green') 
plt.xlabel('Wage Rate (w)') 
plt.ylabel('Max Utility') 
plt.title('Utility as a Function of Wage Rate') 
plt.grid(True) 
plt.legend() 
plt.tight_layout()
plt.show()


#2.7

tau = np.array([37, 42, 45]) 
T = np.array([198.875, 199.5, 196.875]) 

#polynomial interpolation
coefficients = np.polyfit(tau,T,deg=tau.size-1) 
polynomial = np.poly1d(coefficients) 

print(f'polynomial: {polynomial[0]}x^{tau.size-1} + {polynomial[1]}x^{tau.size-2} + {polynomial[2]}')

tau_range = np.linspace(35, 45, 200) 
T_vals = polynomial(tau_range) 

result = minimize_scalar(lambda t: -polynomial(t), bounds=(35, 45), method='bounded') 
tau_opt = result.x 
T_opt = polynomial(tau_opt) 

print(f"τ* = {tau_opt:.4f}")
print(f"T(τ*) = {T_opt:.4f}")

# Plot
plt.plot(tau_range, T_vals, label="Interpolated Revenue Function") 
plt.scatter(tau, T, color='red', label="Data Points") 
plt.xlabel("Tax Rate (τ)") 
plt.ylabel("Tax Revenue T(τ)") 
plt.title("Interpolated Tax Revenue Function") 
plt.grid(True) 
plt.legend()
plt.tight_layout()
plt.show()

#2.9
# Parameters
alpha = 1.0 
eta = 1.5 
m = 3 
N = 10 
NP = 1000 
P_grid = np.linspace(0.1, 3.0, N + 1) 

def D(P):
    return P ** (-eta) 
def marginal_demand(P):
    return -eta * P ** (-eta - 1)

def marginal_cost(q):
    return alpha * np.sqrt(q) + q**2

def firm_q(P):
    def F(q):
  
        MB = P + q * (1 / marginal_demand(P)) * (1 / m) 
        MC = marginal_cost(q) 
        return MB - MC

    q_guess = D(P) / m 
    q_sol, = fsolve(F, q_guess) 
    return max(q_sol, 0)  

def excess_demand(P):
    if P <= 0:
        return 1e6  
    return D(P) - m * spline_q(P)

q_values = np.array([firm_q(P) for P in P_grid])

spline_q = CubicSpline(P_grid, q_values)

P_fine = np.linspace(0.1, 3.0, NP) 
q_interp = spline_q(P_fine) 

P_eq_guess = 1.0
P_eq, = fsolve(excess_demand, P_eq_guess)
Q_eq = D(P_eq)


# Plot demand and individual supply
plt.figure(figsize=(10, 6)) #create a figure
plt.plot(P_fine, D(P_fine), label="Market Demand D(P)") #plot the demand
plt.plot(P_fine, m * q_interp, label="Aggregate Supply (m*q(P))") #plot the aggregate supply
plt.xlabel("Price") #x-axis label
plt.ylabel("Quantity") #y-axis label
plt.title("Market Demand and Aggregate Supply") #title
plt.legend() #legend
plt.grid(True) #grid
plt.savefig('cournot_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Equilibrium price: {P_eq:.4f}, Quantity: {Q_eq:.4f}")
print("Increasing m leads to more competition, pushing price down and quantity up.\nIncreasing α makes costs higher, reducing supply and increasing price.\nIncreasing η makes demand more elastic: flattens demand, leading to more responsive price changes.") 




