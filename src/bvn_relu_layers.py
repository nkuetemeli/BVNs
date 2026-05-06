from dataset import *

from trunk.src.BVNets.cheap_relu import *
from trunk.src.BVNets.cheap_sigmoid import *
from trunk.src.BVNets.modulo import *


class BVUnit:
    """A single quantum processing unit (like a neuron) with structured weight qubits."""
    def __init__(self, id, in_dim, n_qubits_weight, n_qubits_output, start_wire=0, activation='ReLU', activation_params=None):
        self.id = id
        self.in_dim = in_dim
        self.n_qubits_weight = n_qubits_weight
        self.n_qubits_output = n_qubits_output
        self.start_wire = start_wire
        self.activation = activation
        self.activation_params = activation_params or {}

        wire = start_wire
        # Structure weight qubits per input dimension
        self.weight_wires = []
        for i in range(in_dim):
            qubits = list(range(wire, wire + n_qubits_weight))
            self.weight_wires.append(qubits)
            wire += n_qubits_weight

        # Output qubits
        self.out_wires = list(range(wire, wire + n_qubits_output))
        wire += n_qubits_output

        self.end_wire = wire

        self.activation_fn = None

    @property
    def _weight_wires(self):
        return [item for sublist in self.weight_wires for item in sublist]

    def describe(self):
        print(f"\nUnit {self.id}:")
        for idx, q in enumerate(self.weight_wires):
            print(f"  Input {idx} weight qubits: {q}")
        print(f"  Output qubits: {self.out_wires}")
        print(f"  Activation function: {self.activation}")

    def forward(self, input_wires, ancilla_wires):
        sigma_wires = self.out_wires

        if self.activation == 'ReLU':
            x1_wires, x2_wires = input_wires
            y1_wires, y2_wires = self.weight_wires
            threshold = self.activation_params.get('threshold', 2 ** len(self.out_wires))
            print(f'Using threshold {threshold}')
            CheapReLU(x1_wires, x2_wires, y1_wires, y2_wires, [ancilla_wires[0]], sigma_wires)
            self.activation_fn = lambda z: np.where(z < threshold, z, 0).astype(int)

        elif self.activation == 'Sigmoid':
            threshold = int(2 ** (sum(len(i) for i in input_wires) / len(input_wires)))
            threshold = self.activation_params.get('threshold', threshold)
            print(f'Using threshold {threshold}')
            CheapSigmoid(input_wires, self.weight_wires, sigma_wires, ancilla_wires, threshold=threshold)
            self.activation_fn = lambda z: np.where(z < threshold, 1, 0).astype(int)

        if self.activation == 'Modulo':
            modulo = self.activation_params.get('modulo', 2 ** len(self.out_wires))
            print(f'Using modulo {modulo}')
            Modulo(input_wires, self.weight_wires, sigma_wires, ancilla_wires, modulo=modulo)
            self.activation_fn = lambda z: z % modulo


class BVLayer:
    """A quantum layer composed of multiple independent units."""
    def __init__(self, in_dim, n_units, n_qubits_weight, n_qubits_output, start_wire=0,
                 activation='ReLU', activation_params=None):
        self.in_dim = in_dim
        self.n_units = n_units
        self.n_qubits_weight = n_qubits_weight
        self.n_qubits_output = n_qubits_output
        self.start_wire = start_wire
        self.activation = activation
        self.activation = activation
        self.activation_params = activation_params or {}

        self.units = []
        wire = start_wire
        for i in range(n_units):
            unit = BVUnit(
                id=i,
                in_dim=self.in_dim,
                n_qubits_weight=self.n_qubits_weight,
                n_qubits_output=self.n_qubits_output,
                start_wire=wire,
                activation=self.activation,
                activation_params=self.activation_params
            )
            self.units.append(unit)
            wire = unit.end_wire
        self.end_wire = wire

    def total_wires(self):
        return self.end_wire - self.start_wire

    @property
    def _weight_wires(self):
        """Flatten and collect all weight wires from all units in the layer."""
        return [w for unit in self.units for w in unit._weight_wires]

    @property
    def activation_wires(self):
        """Flatten and collect all weight wires from all units in the layer."""
        return [unit.out_wires for unit in self.units]

    @property
    def _activation_wires(self):
        """Flatten and collect all weight wires from all units in the layer."""
        return [w for unit in self.units for w in unit.out_wires]

    def forward(self, input_wires, ancilla_wire):
        """
        Forward pass for the layer.

        :param input_wires: list of wires (or list of lists) representing inputs
        :param unit_fn: a function f(unit, input_wires) that applies the quantum unit operations
        """
        for unit in self.units:
            # unit_fn should implement how the unit acts on its inputs
            # input_wires can be structured as a list of qubit registers per input dimension
            unit.forward(input_wires, ancilla_wire)

    def activation_fn(self, x):
        x_act = np.zeros_like(x)
        for i, (x_i, unit) in enumerate(zip(x, self.units)):
            x_act[i] = unit.activation_fn(x_i)
        return x_act


    def describe(self):
        print(f"\nBVLayer Summary:")
        print(f"  Units: {self.n_units}")
        print(f"  Each unit has {self.in_dim}×{self.n_qubits_weight} weight qubits "
              f"and {self.n_qubits_output} output qubits.")
        print(f"  Total wires in layer: {self.total_wires()}")
        for u in self.units:
            u.describe()



