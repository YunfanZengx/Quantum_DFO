import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_ibm_runtime.fake_provider import FakeTorontoV2
from qiskit.primitives import StatevectorSampler     # ideal_simulator 
from qiskit_ibm_runtime import SamplerV2 as Sampler  # real_backend and fake_backend
from qiskit_ibm_runtime import QiskitRuntimeService

class QAOAMaxCut:
    def __init__(self, graph, p, shots=1000, seed=None, backend_type="ideal_simulator"):
        self.graph = graph
        self.p = p
        self.shots = shots
        self.seed = seed
        self.backend_type = backend_type

        self.sampler = self._load_backend(backend_type)
        self.qc, self.theta_params = self._create_qaoa_circuit(self.graph, self.p)
        self.transpiled_qc = self._transpile_circuit(self.qc)
    
    def _load_backend(self, backend_type):
        if backend_type == "ideal_simulator":
            return StatevectorSampler(default_shots=self.shots, seed=self.seed)
        
        elif backend_type == "fake_backend":
            backend = FakeTorontoV2()
            options = {
                "simulator": {"seed_simulator": self.seed},
                "default_shots": self.shots  
            }
            return Sampler(backend, options=options)
        
        elif backend_type == "real_backend":
            service = QiskitRuntimeService()
            backend = service.least_busy(
                operational=True,       
                simulator=False,        
                min_num_qubits=20       
            )
            options = {"default_shots": self.shots}
            return Sampler(backend, options=options)
        
        else:
            raise ValueError(f"Unsupported backend_type: {backend_type}")

    def _create_qaoa_circuit(self, G, p):
        """
        Create a QAOA circuit with p layers, each with its own gamma and beta parameters
        
        Args:
            G (networkx.Graph): Input graph
            p (int): Number of QAOA layers
        
        Returns:
            qc: quantum circuit
            theta_params: list of gamma parameters, list of beta parameters)
        """
        n_qubits = G.number_of_nodes()
        qc = QuantumCircuit(n_qubits)
        
        # Initialize to uniform superposition
        qc.h(range(n_qubits))
        
        theta_params = [Parameter(f'theta_{i}') for i in range(2*p)]

    # Apply p layers of QAOA
        for i in range(p):
        
            gamma_idx = 2 * i      # theta_0, theta_2, theta_4, ...
            beta_idx = 2 * i + 1   # theta_1, theta_3, theta_5, ...
        
        # Apply Cost Hamiltonian with gamma_i
            for u, v in G.edges():
                    qc.rzz(2 * theta_params[gamma_idx], u, v)
        
        # Apply Mixer Hamiltonian with beta_i  
            for qubit in range(n_qubits):
                qc.rx(2 * theta_params[beta_idx], qubit)
        # Add measurement at the end
        qc.measure_all()
        
        return qc, theta_params

    def _transpile_circuit(self, circuit):
        if isinstance(self.sampler, StatevectorSampler):
            return circuit
        else:
            from qiskit import transpile
            return transpile(
                circuit, 
                backend=self.sampler.backend(), 
                seed_transpiler=self.seed  
            )

    def evaluate_obj(self, params):
        """
        Evaluate QAOA circuit with given parameters
        
        Args:
            params: parameter values, length 2*p, ordered as [theta_0, theta_1, ..., theta_{2p-1}]
            shots: number of sampling shots
            seed: random seed for reproducibility
        
        Returns:
            p_vals: probability distribution of measurement outcomes (array)
            f_vals: cut values for each measured bitstring (array)
            obj: expected cut value (float)
            grad: gradient of the objective function (array)
        """

        # Create parameter binding dictionary
        param_dict = {f'theta_{i}': params[i] for i in range(len(params))}
        sampler = self.sampler
        result = sampler.run([(self.transpiled_qc, param_dict)]).result()
        counts = result[0].data.meas.get_counts()

    # Calculate objective from counts
        n_qubits = self.graph.number_of_nodes()
        all_bitstrings = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
        p_vals = []
        f_vals = []
        obj = 0
    
        
        for bitstring in all_bitstrings:
        # Calculate probability
            count = counts.get(bitstring, 0)
            prob = count / self.shots
            p_vals.append(prob)
        
        # Calculate cut value
            cut_value = 0
            for u, v in self.graph.edges():
                if bitstring[u] != bitstring[v]:
                    cut_value += 1
            f_vals.append(cut_value)
        
        # Accumulate expected value
            obj += prob * cut_value
    
        return np.array(p_vals), np.array(f_vals), obj
        
    def finite_difference_gradient(self, params):
        """
        Calculate gradient using parameter shift rule
        
        Args:
            params: current parameter values
            shots, seed: same as evaluate function
        
        Returns:
            gradient: array of gradients for each parameter
        """
        gradients = []
        shift = 0.01  # shift amount for our QAOA gates
        
        for i in range(len(params)):
            # Create shifted parameter arrays
            params_plus = params.copy()
            params_minus = params.copy()
            
            params_plus[i] += shift
            params_minus[i] -= shift
            
            # Evaluate at shifted points
            _, _, obj_plus = self.evaluate_obj(params_plus)
            _, _, obj_minus = self.evaluate_obj(params_minus)
            
            # Calculate gradient for parameter i
            grad_i = (obj_plus - obj_minus)/(2*shift)
            gradients.append(grad_i)
        
        return np.array(gradients)

if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Set up K5 graph with p=2
    G = nx.complete_graph(5)
    p = 2
    
    # Create QAOAMaxCut instance
    qaoa = QAOAMaxCut(graph=G, p=p, backend_type="fake_backend", shots=1000, seed=42)
    
    # Random initial parameters
    np.random.seed(42)
    initial_params = np.random.uniform(0, 2*np.pi, 2*p)
    
    print("=== QAOA MaxCut Gradient Descent Test ===")
    print(f"Initial parameters: {initial_params}")
    
    # Evaluate initial objective
    _, _, initial_obj = qaoa.evaluate_obj(initial_params)
    print(f"Initial objective: {initial_obj:.6f}")
    
    # Simple gradient ascent
    params = initial_params.copy()
    lr = 0.01
    steps = 150
    history = [initial_obj]
    
    for step in range(steps):
        grad = qaoa.finite_difference_gradient(params)
        params = params + lr * grad
        _, _, obj = qaoa.evaluate_obj(params)
        history.append(obj)
        if step % 10 == 0:
            print(f"Step {step}: obj = {obj:.6f}")
    
    print(f"Final objective: {history[-1]:.6f}")
  
    # Plot results
    plt.plot(history)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.title('QAOA Gradient Descent on K5')
    plt.show()