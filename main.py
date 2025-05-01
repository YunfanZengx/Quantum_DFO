import sys
import numpy as np
np.random.seed(43)

# Make sure Python can find your modules
sys.path.append('/mnt/data')

# 1) QAOA setup from demo.py
from demo import create_qaoa_circuit, get_objective, G

# 2) Your FDLM optimizer
from DFO_QN_test1 import FDLM, Parameters


# 3) Build the QAOA circuit & parameter list (must match demo.py’s p)
p = 2
qaoa_circuit, gamma_params, beta_params = create_qaoa_circuit(G, p)
params_names = gamma_params + beta_params

# 4) Wrap the demo objective into a scalar function
def qaoa_objective(x: np.ndarray) -> float:
    # FDLM minimizes, so get_objective already returns negative MaxCut
    return get_objective(x, qaoa_circuit, params_names, G, p)

# 5) Pick an initial point of length 2*p
x0 = np.random.uniform(0, np.pi, 2 * p)

# 6) Run FDLM and print results
sol, final_f, best_f = FDLM(qaoa_objective, x0, params=Parameters)
print("Final parameters (x):", sol)
print(f"Objective at final x: {final_f:.6e}")
print(f"Best objective observed: {best_f:.6e}")
