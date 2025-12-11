from utils import BarsAndStripes, MMD_Gauss_Mix

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml


def main():
    """Debugging the MMD loss..."""

    jax.config.update("jax_enable_x64", True)

    ds = BarsAndStripes(n=3)

    n_shots = 10
    n_qubits = ds.size
    n_layers = 2

    dev = qml.device("default.qubit", wires=n_qubits)

    wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = jnp.random.random(size=wshape)

    @qml.set_shots(n_shots)
    @qml.qnode(dev)
    def pqc(weights):
        qml.StronglyEntanglingLayers(
            weights=weights, ranges=[1] * n_layers, wires=range(n_qubits)
        )
        return qml.sample()

    jit_circuit = jax.jit(pqc)

    scales = jnp.array([0.25, 0.5, 1])

    mmd = MMD_Gauss_Mix(
        scales=scales,
        circuit=jit_circuit,
        target_dist=ds.sample_integer,
        n_shots=n_shots,
    )

    grads = mmd.compute_gradient(weights)
    assert grads.size == weights.size

    print(weights.shape)
    print(grads)

    for i, _ in enumerate(weights):
        print(i)


if __name__ == "__main__":
    main()
