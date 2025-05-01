import numpy as np

# ---------------- Algorithm Parameters ----------------
class Parameters:
    # Noise and step size parameters
    NOISE_LEVEL = 1e-6    # objective function noise bound
    FIXED_H = 1e-4        # finite difference step size
    
    # L-BFGS parameters
    MAX_MEMORY = 5        # maximum number of (s,y) pairs to store
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
    MAX_ITER = 90        # maximum number of iterations
    TOL = 1e-6           # convergence tolerance
    VERBOSE = True       # print iteration information

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function with bounded noise"""
    noise = np.random.uniform(-Parameters.NOISE_LEVEL, Parameters.NOISE_LEVEL)
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2 + noise

def ecnoise(*_args, **_kwargs) -> float:
    """Return estimated noise level"""
    return Parameters.NOISE_LEVEL

def estimate_gradient(f, x: np.ndarray, eps_f: float):
    """Forward finite difference gradient estimation"""
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

def lbfgs_two_loop_recursion(g, s_list, y_list, gamma=1.0):
    """L-BFGS two-loop recursion with damping"""
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

def line_search(f, xk, fk, gradk, dk, eps_f, params=Parameters):
    """Armijo-only line search"""
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

def recovery(f, xk, fk, dk, eps_f, best_x, best_f, h_old, params=Parameters):
    """Recovery procedure when line search fails"""
    eps_dk = ecnoise()
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

def FDLM(f, x0, params=Parameters):
    """Main optimization routine with damping L-BFGS"""
    x = x0.copy()
    fx = f(x)
    eps_f = ecnoise()
    gradk, best_x, best_f, h = estimate_gradient(f, x, eps_f)
    s_list, y_list = [], []
    
    for k in range(params.MAX_ITER):
        if params.VERBOSE:
            print(f"\n------------------ Iteration {k + 1} ----------------")
            print(f"Current x = {x}, f(x) = {fx:.6e}")
        
        # Compute search direction
        if s_list:
            s_k = s_list[-1]
            y_k = y_list[-1]
            ss = np.dot(s_k, s_k)
            sy = np.dot(s_k, y_k)
            
            # Apply damping if necessary
            if sy <= params.DAMPING_THETA * ss:
                gamma = (params.DAMPING_THETA * ss - sy) / (ss - sy)
                y_hat = gamma * s_k + (1 - gamma) * y_k
                y_list[-1] = y_hat
            
            sy = np.dot(s_list[-1], y_list[-1]) + 1e-16
            yy = np.dot(y_list[-1], y_list[-1])
            gamma = sy / yy if 0 < sy < 1e10 else 1.0
            dk = -lbfgs_two_loop_recursion(gradk, s_list, y_list, gamma)
            
            if sy <= params.DAMPING_THETA * ss:
                y_list[-1] = y_k
        else:
            dk = -gradk
        
        # Line search and update
        x_new, f_new, alpha, ls_fail = line_search(f, x, fx, gradk, dk, eps_f, params)
        
        if ls_fail:
            print(f"Line search failed, trying recovery...")
            x_new, f_new, h, _ = recovery(f, x, fx, dk, eps_f, best_x, best_f, h, params)
        
        # Update L-BFGS vectors
        s_k = x_new - x
        x, fx = x_new, f_new
        grad_new, best_x2, best_f2, h = estimate_gradient(f, x, eps_f)
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
        if best_f2 < best_f:
            best_f, best_x = best_f2, best_x2
            
        if np.linalg.norm(gradk) < params.TOL:
            if params.VERBOSE:
                print(f"\nConverged: ‖g‖ < {params.TOL}")
            break
            
    return x, fx, best_f

if __name__ == "__main__":
    np.random.seed(0)
    x0 = np.array([-1.2, 1.0])
    
    # Optional: customize parameters
    custom_params = Parameters()
    custom_params.MAX_ITER = 100
    custom_params.TOL = 1e-8
    
    sol, final_f, best_f = FDLM(rosenbrock, x0, params=custom_params)
    print(f"\nFinal solution: {sol}")
    print(f"Final objective: {final_f:.6e}")
    print(f"Best objective found: {best_f:.6e}")