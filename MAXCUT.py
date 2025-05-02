import numpy as np
import networkx as nx
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_optimization.applications.max_cut import Maxcut
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class QAOAMaxCut:
    def __init__(
        self,
        graph: nx.Graph,
        p: int,
        backend_options: dict
    ):
        """
        graph: NetworkX graph or adjacency matrix
        p: Number of QAOA layers
        backend_options: {
            'type':      'ideal_simulator', 'fake_backend_simulator','real_backend_simulator' , or 'real_backend',   # ideal simulator or real quantum device
            'name':      'aer_simulator' or 'ibm_xyz'                                                     # backend name for AerSimulator or IBM device
        }
        """
        self.graph = graph
        self.p = p
        self.backend_options = backend_options

        # Load backend and noise model
        self.backend, self.noise_model = self._load_backend(backend_options)

        # Prepare cost operator, offset, and ansatz
        self.qubit_op, self.offset, self.ansatz = self._prepare_ansatz()

    def _load_backend(self, opts: dict):
        """
        Private method: load IBM Quantum account and configure backend and noise model
        Returns: (backend, noise_model)
        """
        self.backend_type = opts.get('type', 'ideal').lower()
        self.backend_name = opts.get('name', 'aer_simulator')

        # Default to noiseless local simulator
        backend = AerSimulator()
        noise_model = None


        if self.backend_type == 'ideal_simulator':
            
            return backend,noise_model
            
        elif self.backend_type == 'fake_backend_simulator':

            backend = FakeManilaV2()
            noise_model = NoiseModel.from_backend(backend)
            
        else:
            try:
                service = QiskitRuntimeService()
                print("Info: IBM Quantum account loaded; ready for cloud backends.")
            except Exception as e:
                raise RuntimeError(f"Warning: Failed to load IBM Quantum Service ({e}); please use local simulator.")

            if self.backend_type == 'real_backend_simulator':
                real = service.backend(self.backend_name)
                noise_model = NoiseModel.from_backend(real)
                cm = real.configuration().coupling_map
                bg = real.basis_gates
                backend = AerSimulator(
                                    noise_model=noise_model,
                                    coupling_map=cm,
                                    basis_gates=bg
                                        )
            elif self.backend_type == 'real_backend':
                backend = service.backend(self.backend_name)
                noise_model = None

            else:
                raise ValueError(f"Unknown backend type: {self.backend_type}")
        
        return backend,noise_model

    
    def _prepare_ansatz(self):
        """
        Private method: construct the cost operator, offset, and QAOA ansatz
        Returns: (qubit_op, offset, ansatz)
        """
        # Build adjacency matrix with edge weights
        w_mat = nx.to_numpy_array(self.graph, weight='weight')

        # Map to Ising Hamiltonian
        maxcut = Maxcut(w_mat)
        qubit_op, offset = maxcut.to_quadratic_program().to_ising()

        # Build QAOA ansatz
        ansatz = QAOAAnsatz(cost_operator=qubit_op, reps=self.p)
        
        
        return qubit_op, offset, ansatz

    def evaluate(
        self,
        params: np.ndarray,
        shots: int = 1024,
        seed: int = None
    ) -> float:
        """
        Evaluate QAOA cost for given parameters using SamplerV2.

        params: array of length 2*p in the order defined by ansatz.ordered_parameters
        shots:  number of measurement shots
        seed:   random seed for simulator
        Returns: expected cost 
        """

        # Construct the Quantum Circuit
        qc = self.ansatz.assign_parameters(params, inplace=False)
        qc = qc.decompose().decompose()
        qc.measure_all()


        # Prepare sampler Options 
        sampler_opts = {"default_shots": shots}

        # Only Simulator needs simulator options
        if self.backend_type == 'ideal_simulator':
            sim_opts = {}
            if seed is not None:
                sim_opts["seed_simulator"] = seed
                sampler_opts["simulator"] = sim_opts
        
        elif self.backend_type == 'fake_backend_simulator':
            sim_opts = {}
            if seed is not None:
                sim_opts['seed_simulator'] = seed
            sampler_opts['simulator'] = sim_opts  
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
            qc = pm.run(qc)
            
        elif self.backend_type == 'real_backend_simulator':
            sim_opts = {}
            if seed is not None:
                sim_opts['seed_simulator'] = seed
            if self.noise_model is not None:
                sim_opts.update({
                    'noise_model': self.noise_model,
                    'coupling_map': self.backend.configuration().coupling_map,
                    'basis_gates': self.noise_model.basis_gates
                })
            sampler_opts['simulator'] = sim_opts

        elif self.backend_type == 'real_backend':
            pass
        
        sampler = Sampler(mode=self.backend,options=sampler_opts)
        job = sampler.run([qc])
        results = job.result()
        pub_res = results[0]                        
        counts = pub_res.join_data().get_counts()


        exp_val = 0.0
        for bitstring, count in counts.items():
            # Reverse bitstring to match node ordering
            x = [int(bit) for bit in bitstring[::-1]]
            # Count edges cut by this bitstring
            for i, j, weight in self.graph.edges(data='weight', default=1.0):
                if x[i] != x[j]:
                    exp_val += weight * (count / shots)
        # 5. Return cost to minimize
        return float(exp_val)
if __name__ == "__main__":
    # 1. Create a graph Instance
    G = nx.Graph()
    G.add_edge(0, 1)

    # 2. Create a QAOAMaxCut instance (You can change the backend_options)
    instance1 = QAOAMaxCut(
        graph=G,
        p=1,
        backend_options={'type': 'ideal_simulator', 'name': 'aer_simulator'}
    )

    print("Backend:", instance1.backend)

    # 3. Evaluate the cost at γ=β=0 (The theoretical value should be close to 1.0)
    params = np.zeros(2 * instance1.p)   # [γ0, β0] = [0, 0]
    loss1= instance1.evaluate(params)
    print(f"Cost at γ=β=0 (default shots, seed=42): {loss1:.4f}")

    # 4. Evaluate the cost at γ=β=0 (shots=200, seed=7)
    loss2 = instance1.evaluate(params, shots=200, seed=7)
    print(f"Cost at γ=β=0 (shots=200, seed=7):       {loss2:.4f}")

    G2 = nx.Graph()
    G2.add_edge(0, 1)
    G2.add_edge(1, 2)


    instance2 = QAOAMaxCut(
        graph=G2,
        p=1,
        backend_options={'type': 'fake_backend_simulator', 'name': 'aer_simulator'}
    )

    print("Backend:", instance2.backend)

    params = np.zeros(2 * instance2.p)   # [γ0, β0] = [0, 0]
    loss3 = instance2.evaluate(params)
    print(f"Cost at γ=β=0 (default shots, seed=42): {loss3:.4f}")
