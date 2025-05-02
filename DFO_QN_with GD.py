import numpy as np
from typing import Callable, Tuple, List, Optional
from abc import ABC, abstractmethod

class Parameters:
    # Noise and step size parameters
    NOISE_LEVEL = 0    # objective function noise bound
    FIXED_H = 1e-8        # finite difference step size

    # L-BFGS parameters
    MAX_MEMORY = 20        # maximum number of (s,y) pairs to store
    DAMPING_THETA = 0.1   # damping parameter for s^T y <= theta * ||s||^2

    # Line search parameters
    C1 = 1e-4            # Armijo condition parameter
    ALPHA_INIT = 1.0     # initial step size
    TAU = 0.7            # step size reduction factor
    MIN_STEP = 1e-6      # minimum allowed step size

    # Recovery parameters
    GAMMA1 = 0.5         # lower bound for step size ratio
    GAMMA2 = 2.0         # upper bound for step size ratio
    FECN = 4             # number of function evaluations for noise estimation

    # Algorithm parameters
    MAX_ITER = 200       # increased max iterations for higher dimension
    TOL = 1e-6           # convergence tolerance
    VERBOSE = True       # print iteration information

class ObjectiveFunction(ABC):
    def __init__(self, noise_level: float = 1e-8):
        self.noise_level = noise_level
        self.eval_count = 0

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def true_value(self, x: np.ndarray) -> float:
        pass

    def noise(self) -> float:
        return self.noise_level

class Rosenbrock5D(ObjectiveFunction):
    def __call__(self, x: np.ndarray) -> float:
        # count this evaluation
        self.eval_count += 1
        noise = np.random.uniform(-self.noise_level, self.noise_level)
        return self.true_value(x) + noise

    def true_value(self, x: np.ndarray) -> float:
        """5D Rosenbrock function"""
        return sum(100.0*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

class Rosenbrock4D(ObjectiveFunction):
    def __call__(self, x: np.ndarray) -> float:
        self.eval_count += 1
        noise = np.random.uniform(-self.noise_level, self.noise_level)
        return self.true_value(x) + noise

    def true_value(self, x: np.ndarray) -> float:
        """4D Rosenbrock function"""
        return sum(100.0*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

class Rosenbrock3D(ObjectiveFunction):
    def __call__(self, x: np.ndarray) -> float:
        self.eval_count += 1
        noise = np.random.uniform(-self.noise_level, self.noise_level)
        return self.true_value(x) + noise

    def true_value(self, x: np.ndarray) -> float:
        """3D Rosenbrock function"""
        return sum(100.0*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))


def estimate_gradient(f: ObjectiveFunction, x: np.ndarray, eps_f: float):
    n, h = len(x), Parameters.FIXED_H
    g = np.zeros(n)
    fx = f(x)
    best_fx = fx
    best_x = x.copy()

    for i in range(n):
        x_fd = x.copy()
        x_fd[i] += h
        f_fd = f(x_fd)
        g[i] = (f_fd - fx) / h
        if f_fd < best_fx:
            best_fx, best_x = f_fd, x_fd
    return g, best_x, best_fx, h

def lbfgs_two_loop_recursion(g: np.ndarray, s_list: List[np.ndarray], y_list: List[np.ndarray], gamma: float = 1.0) -> np.ndarray:
    q = g.copy()
    alpha_vals = []
    
    for s, y in reversed(list(zip(s_list, y_list))):
        sy = np.dot(s, y)
        alpha = 0.0 if abs(sy) < 1e-16 else np.dot(s, q) / sy
        alpha_vals.append(alpha)
        q -= alpha * y
    
    z = gamma * q
    
    for (s, y), alpha in zip(zip(s_list, y_list), reversed(alpha_vals)):
        sy = np.dot(s, y)
        beta = 0.0 if abs(sy) < 1e-16 else np.dot(y, z) / sy
        z += (alpha - beta) * s
    return z

def line_search(f: ObjectiveFunction, xk: np.ndarray, fk: float, gradk: np.ndarray, 
                dk: np.ndarray, eps_f: float, params: Parameters = Parameters) -> Tuple[np.ndarray, float, float, bool]:
    alpha = params.ALPHA_INIT
    dir_deriv = np.dot(gradk, dk)
    
    if not np.all(np.isfinite(dk)):
        return xk, fk, 0.0, True
    
    while alpha > params.MIN_STEP:
        x_new = xk + alpha * dk
        f_new = f(x_new)
        
        if f_new <= fk + params.C1 * alpha * dir_deriv + 2 * eps_f:
            return x_new, f_new, alpha, False
            
        alpha *= params.TAU
    
    return xk, fk, alpha, True

def recovery(f: ObjectiveFunction, xk: np.ndarray, fk: float, dk: np.ndarray, eps_f: float, 
             best_x: np.ndarray, best_f: float, h_old: float, params: Parameters = Parameters) -> Tuple[np.ndarray, float, float, int]:
    h_new = params.FIXED_H
    trec = params.FECN
    
    ratio = h_new / (h_old + 1e-16)
    if ratio < params.GAMMA1 or ratio > params.GAMMA2:
        return xk, fk, h_old, trec

    norm_dk = np.linalg.norm(dk) + 1e-16
    xh = xk + (h_old / norm_dk) * dk
    fh = f(xh)
    trec += 1
    
    if fh <= fk + params.C1 * (h_old / norm_dk) * np.dot(dk, dk):
        return xh, fh, h_old, trec
    if fh < fk and fh < best_f:
        return xh, fh, h_old, trec
    if best_f < fk and best_f < fh:
        return best_x, best_f, h_old, trec
    return xk, fk, h_old, trec

def FDLM(objective: ObjectiveFunction, x0: np.ndarray, params: Parameters = Parameters) -> Tuple[np.ndarray, float, float, float]:
    x = x0.copy()
    fx = objective(x)
    eps_f = objective.noise()
    gradk, best_x, best_f, h = estimate_gradient(objective, x, eps_f)
    s_list, y_list = [], []
    best_true = objective.true_value(best_x)
    
    for k in range(params.MAX_ITER):
        if params.VERBOSE:
            print(f"\n------------------ Iteration {k + 1} ----------------")
            print(f"True f(x) = {objective.true_value(x):.6e}")
        
        if s_list:
            s_k = s_list[-1]
            y_k = y_list[-1]
            ss = np.dot(s_k, s_k)
            sy = np.dot(s_k, y_k)
            
            if sy <= params.DAMPING_THETA * ss:
                gamma = (params.DAMPING_THETA * ss - sy) / (ss - sy)
                print(ss, sy, gamma)   
                y_hat = gamma * s_k + (1 - gamma) * y_k
                y_list[-1] = y_hat
            
            sy = np.dot(s_list[-1], y_list[-1]) + 1e-16
            yy = np.dot(y_list[-1], y_list[-1])
            gamma = sy / yy if 0 < sy < 1e10 else 1.0
            dk = -lbfgs_two_loop_recursion(gradk, s_list, y_list, gamma)
            print(f"sy = {sy}, ss = {ss}, yy = {yy}, gamma = {gamma}")
            if sy <= params.DAMPING_THETA * ss:
                y_list[-1] = y_k
        else:
            dk = -gradk
        
        x_new, f_new, alpha, ls_fail = line_search(objective, x, fx, gradk, dk, eps_f, params)
        
        # remove if you don't want to use recovery
        if ls_fail:
            x_new, f_new, h, _ = recovery(objective, x, fx, dk, eps_f, best_x, best_f, h, params)
        
        # compute step
        s_k = x_new - x
        if np.linalg.norm(s_k) < 1e-14:  # Near-zero step
            if params.VERBOSE:
                print("Step size too small - Restarting L-BFGS with gradient descent")
            s_list.clear()
            y_list.clear()
            dk = -gradk
            x_new, f_new, alpha, _ = line_search(objective, x, fx, gradk, dk, eps_f, params)
            s_k = x_new - x
            x, fx = x_new, f_new
            grad_new, _, _, h = estimate_gradient(objective, x, eps_f)
            gradk = grad_new
            continue

        x, fx = x_new, f_new
        true_val = objective.true_value(x)
        if true_val < best_true:
            best_true = true_val
            best_x = x.copy()
        
        grad_new, _, _, h = estimate_gradient(objective, x, eps_f)
        y_k = grad_new - gradk
        
        ss = np.dot(s_k, s_k)
        sy = np.dot(s_k, y_k)
        
        if sy <= params.DAMPING_THETA * ss:
            gamma = (params.DAMPING_THETA * ss - sy) / (ss - sy)
            y_hat = gamma * s_k + (1 - gamma) * y_k
            s_list.append(s_k)
            y_list.append(y_hat)
        else:
            s_list.append(s_k)
            y_list.append(y_k)
        
        if len(s_list) > params.MAX_MEMORY:
            s_list.pop(0)
            y_list.pop(0)
        
        gradk = grad_new
        if np.linalg.norm(gradk) < params.TOL:
            if params.VERBOSE:
                print(f"\nConverged: ‖g‖ < {params.TOL}")
            break
            
    return best_x, fx, objective.true_value(best_x), best_true

if __name__ == "__main__":
    np.random.seed(42)
    
    # Initialize 3D problem
    n_dim = 3
    x0 = np.array([-1.2] + [1.0] * (n_dim-1))  # Standard initial point for Rosenbrock
    
    # Create objective function
    rosen = Rosenbrock3D(noise_level=1e-8)
    
    # Optional: customize parameters
    custom_params = Parameters()
    custom_params.MAX_ITER = 300  # Increased for higher dimension
    custom_params.TOL = 1e-6
    
    sol, final_f, final_true, best_true = FDLM(rosen, x0, params=custom_params)
    print("\nOptimization Results:")
    print(f"Initial point: {x0}")
    print(f"Final solution: {sol}")
    print(f"Final true objective: {final_true:.6e}")
    print(f"Best true objective found: {best_true:.6e}")
    print(f"Distance to optimum: {np.linalg.norm(sol - 1.0):.6e}")
    print(f"Number of function evaluations: {rosen.eval_count}")
