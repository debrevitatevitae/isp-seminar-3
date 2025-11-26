import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #iSP Seminar 3
    ## Parametrized quantum circuits and variational quantum machine learning

    ###Contents
    1. [A primer on quantum computation](#a primer on quantum computation)

    ### Disclaimer
    We will talk here about **parametrized quantum circuits**. They allow to build relatively compact quantum models, that are runnable on **near-term quantum hardware** within a **classical-quantum training loop**.

    This is nowadays the most common form of quantum machine learning (QML), especially in applications. But QML is not limited to this! For an interesting and critical primer on the subject, see [^1].

    [^1]: S.Y. Chuang, M. Cerezo *A Primer on Quantum Machine Learning*, arXiv:2511.15969
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A primer on quantum computation

    #### One-qubit states
    The basic unit of quantum computation is the **qubit**. Opposite to a bit, which is only allowed to be in one of two states, e.g. $|0\rangle$ and $|1\rangle$, a qubit $|\psi\rangle$ can be in any **complex convex combination** of $|0\rangle$ and $|1\rangle$,

    $$
    |\psi\rangle = \alpha_0|0\rangle + \alpha_1|1\rangle =
    \begin{pmatrix}
        \alpha_0\\
        \alpha_1
    \end{pmatrix},
    $$

    where $\alpha_0, \alpha_1 \in \mathbb{C}$. The last expression of $|\psi\rangle$ is also known as its **state vector** form. The amplitudes of the state vector must obey the following constraint:

    $$
    |\alpha_0|^2 + |\alpha_1|^2 = 1.
    $$

    Therefore, the squared amplitudes of $|\psi\rangle$ are **probabilities** of the state being in the $|0\rangle$ state or in the $|1\rangle$ state respectively. Until the qubit remains unobserved, it is said to be in a **superposition** of $|0\rangle$ and $|1\rangle$.

    The physical meanings of $|\alpha_0|^2, |\alpha_1|^2$ is that they are the probabilities of $|\psi\rangle$ to collapse to either $|0\rangle$ or $|1\rangle$ upon **measurement**. More on measurement later...

    ####The Bloch sphere
    One-qubit quantum states have an intuitive geometrical representation via the **Bloch sphere**. We see below the Bloch sphere of
    - The two 1-bit states.
    - The 1 qubit states (interactive.)
    """)
    return


@app.cell
def _(mo):
    import qutip
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    b = qutip.Bloch(axes=ax)
    b.make_sphere()

    up = qutip.basis(2, 0)
    b.add_states(up)
    b.render()

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    b = qutip.Bloch(axes=ax)
    b.make_sphere()

    down = qutip.basis(2, 1)
    b.add_states(down)
    b.render()

    mo.mpl.interactive(fig)
    return (qutip,)


@app.cell
def _(mo):
    import numpy as np

    theta = mo.ui.slider(0, np.pi, step=0.05, label="theta")
    phi = mo.ui.slider(0, 2*np.pi, step=0.05, label="phi")

    theta, phi
    return np, phi, theta


@app.cell
def _(mo, np, phi, qutip, theta):
    def qubit_bloch_plot(theta, phi):
        b = qutip.Bloch()
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        b.add_vectors([x, y, z])

        # Don't call b.show()
        b.make_sphere()   # ensure it's rendered into b.fig

        return b.fig

    mo.mpl.interactive(qubit_bloch_plot(theta.value, phi.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####Multiple qubit states
    The easiest way in which multiple qubits are combined is when they are **uncorrellated**. For example, if two qubits _can be described separately_ as

    $$
    |\psi_0\rangle =
    \begin{pmatrix}
        \alpha_{0}\\
        \alpha_{1}
    \end{pmatrix},\qquad
    |\psi_1\rangle =
    \begin{pmatrix}
        \beta_{0}\\
        \beta_{1}
    \end{pmatrix},
    $$

    then the compound state-vector is simply the tensor product of the two separate 1-qubit state-vectors,

    $$
    |\psi_\rangle = |\psi_0\rangle \otimes |\psi_1\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle =
    \begin{pmatrix}
        \alpha_{00}\\
        \alpha_{01}\\
        \alpha_{10}\\
        \alpha_{11}
    \end{pmatrix}.
    $$

    Again, the normalization constraint must hold:

    $$
    \sum_{i\in\{0,1\}^2}|\alpha_i|^2=1.
    $$

    But things get more interesting than this! This happens when two states are correllated or **entangled**, i.e.

    $$
    \exist\; |\psi\rangle\neq\bigotimes_{i=1}^n|\psi_i\rangle.
    $$

    For example, consider one of the _Bell states_,

    $$
    |\psi\rangle = \frac{1}{\sqrt{2}}( |00\rangle + |11\rangle ).
    $$

    The two qubits are completely correllated, since measuring one of them automatically determines the other one.

    > **Note**
    >
    > As you can see, 2 qubits give you a state-vector in a 4-dimensional Hilbert space (basis states: $|00\rangle$, $|01\rangle$, $|10\rangle$, $|11\rangle$). More in general, n qubits live in a $2^n$-dimensional Hilbert space.
    >
    > The fact that qubits are in a superposition of an exponential number of classical basis states is one of the ingredients of **quantum advantage** in certain algorithms, but it's sometimes misinterpreted as "qubits contain an exponential amount of classical information". This is **false**, because we are not able to access the amplitudes $\alpha_i$ during the quantum computation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####Operations on qubits
    Quantum algorithms work by applying operations on qubits. In quantum computation, the only operations allowed are linear and **unitary**, that is

    $$
    U^{-1}=U^\dagger,\quad U\in \mathbb{C}^{2^n\times2^n}.
    $$

    Since they act linearly on the state-vectors, unitary operations are **matrices**.

    The fact that operations are unitaries allows to preserve the unit norm of the state-vectors.

    $$
    |\hat{\psi}\rangle = U|\psi\rangle \rightarrow \langle\hat{\psi}|\hat{\psi}\rangle = \langle\psi|U^{\dagger}U|\psi\rangle = \langle\psi|\psi\rangle = 1.
    $$

    Let's look at some of the most typical 1 and 2 qubit operations or **gates** and their matrix representation. They will recur in almost every quantum circuit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _src = "./figures/gate_ops_table.jpg"
    mo.image(src=_src, rounded=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####Measurements
    Measurements are what allows to extract information from the quantum system. They are the only non-linear non-unitary operations allowed on qubits.

    #####Example: measurement in the computational basis
    For the state

    $$
    |\psi\rangle = \frac{|0\rangle+|1\rangle}{\sqrt{2}}
    $$

    we want to measure the observable

    $$
    Z = |0\rangle\langle0| - |1\rangle\langle1| = P_0 - P_1.
    $$

    The probability of obtaining $|0\rangle$ is

    $$
    p_0 = \langle\psi|P_0|\psi\rangle = \frac{1}{2},
    $$

    and the probability of obtaining $|1\rangle$ is

    $$
    p_1 = \langle\psi|P_1|\psi\rangle = \frac{1}{2}.
    $$

    The state after measurements can be

    $$
    \frac{P_0|\psi\rangle}{\sqrt{p_0}} = |0\rangle,
    $$

    $$
    \frac{P_1|\psi\rangle}{\sqrt{p_1}} = |1\rangle.
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####Quantum circuits
    Quantum circuits are graphical representations of the qubits and the operations on them. Here is an example...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from functools import partial
    import pennylane as qml

    from pennylane import numpy as pnp

    n_wires = 5
    n_param_gates = 8
    n_layers = 6

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    @partial(qml.transforms.decompose, max_expansion=1)
    def rnd_circuit():
        shape = qml.RandomLayers.shape(n_layers=n_layers, n_rotations=n_param_gates)
        weights = pnp.random.random(size=shape)
        qml.RandomLayers(weights=weights, wires=range(n_wires))
        return qml.expval(qml.Z(0))

    fig_gen_circ, _ = qml.draw_mpl(rnd_circuit, decimals = 2, style = "pennylane")()
    mo.mpl.interactive(fig_gen_circ)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Quantum circuits show how an algorithm is implemented with a given alphabet of operations, e.g. [those in the previous section](#operations-on-qubits). Note that they are not the lowest level implementation of the algorithm and that they generally need to be compiled to the quantum hardware, since

    1. The circuit's alphabet might be different than the available operations on hardware.
    2. The qubit connectivity of the hardware might be limited and we need to re-route some operation.

    Note also how some gates accept parameters. If these parameters are left free, the circuits can learn (up to a certain extent) specific tasks. This is the core idea of variational quantum algorithms (VQAs)[^1] and what we are going to explore next...

    [^1]: Variational Quantum Algorithms, Quantum Machine Learning and Parametrized Quantum Circuits are, to a great extent, synonyms.
    """)
    return


if __name__ == "__main__":
    app.run()
