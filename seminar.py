import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
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


@app.cell
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
