from jax import numpy as jnp
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


class BarsAndStripes:
    def __init__(self, n: int) -> None:
        """Dataset of binary-pixel images consisting of either bars or stripes.
        Can be used for unsupervised learning.

        Args:
            n(int): number of pixels.
        """

        self.n_pixels = n
        self.size = n**2

        # build the dataset.
        bitstrings = [list(np.binary_repr(i, n))[::-1] for i in range(2**n)]
        bitstrings = np.array(bitstrings, dtype=int)

        stripes = bitstrings.copy()
        stripes = np.repeat(stripes, n, 0)
        stripes = stripes.reshape(2**n, n * n)

        bars = bitstrings.copy()
        bars = bars.reshape(2**n * n, 1)
        bars = np.repeat(bars, n, 1)
        bars = bars.reshape(2**n, n * n)

        self.data = np.vstack(
            (stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]])
        )

        self.bitstrings = []
        self.int_labels = []
        for d in self.data:
            self.bitstrings += ["".join(str(int(i)) for i in d)]
            self.int_labels += [int(self.bitstrings[-1], 2)]

        self.probs = np.zeros(2**self.size)
        self.probs[self.int_labels] = 1 / len(self.data)

    def sample_integer(self, n) -> np.ndarray:
        return np.random.choice(self.int_labels, size=n, p=self.probs[self.int_labels])

    def plot_sample(self, sample_id: int) -> matplotlib.figure.Figure:
        """Plots one of the bar/stripes images.

        Args:
            sample_id (int): id of the image to plot.
        """
        n = self.n_pixels
        sample = self.data[sample_id].reshape(n, n)

        print(f"\nSample bitstring: {''.join(np.array(sample.flatten(), dtype='str'))}")

        plt.figure(figsize=(2, 2))
        plt.imshow(sample, cmap="gray", vmin=0, vmax=1)
        plt.grid(color="gray", linewidth=2)
        plt.xticks([])
        plt.yticks([])

        for i in range(n):
            for j in range(n):
                plt.text(
                    i,
                    j,
                    sample[j][i],
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=12,
                )

    def plot_dataset(self) -> matplotlib.figure.Figure:
        """Plots the entire set of images."""
        n = self.n_pixels

        plt.figure(figsize=(4, 4))
        j = 1
        for i in self.data:
            plt.subplot(4, 4, j)
            j += 1
            plt.imshow(np.reshape(i, (n, n)), cmap="gray", vmin=0, vmax=1)
            plt.xticks([])
            plt.yticks([])

    def plot_data_dist(self) -> matplotlib.figure.Figure:
        """Plots the distribution of the bitstrings.
        Each bitstring has probability equal to 1/size(image_data) if it corresponds
        to a bar or stripes image, zero otherwise.
        """
        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(2**self.size), self.probs, width=2.0, label=r"$\pi(x)$")
        plt.xticks(self.int_labels, self.bitstrings, rotation=80)

        plt.xlabel("Samples")
        plt.ylabel("Prob. Distribution")
        plt.legend(loc="upper right")
        plt.subplots_adjust(bottom=0.3)


class MMD_Gauss_Mix:
    def __init__(self, scales, circuit, target_dist, n_shots):
        self.scales = scales
        self.gammas = 1 / (2 * scales)
        self.circuit = circuit
        self.target_dist = target_dist
        self.n_shots = n_shots

        self.weights = None

    def get_circuit_samples(self, weights):
        return jnp.array(
            [
                jnp.dot(pred, 2 ** jnp.arange(pred.size - 1, -1, -1))
                for pred in self.circuit(weights)
            ]
        )

    def get_target_samples(self):
        return jnp.array(self.target_dist(self.n_shots))

    def compute_kernel(self, x, y):
        k_xy = 0.0
        for gamma in self.gammas:
            k_xy += jnp.exp(-gamma * (x - y) ** 2)

        return k_xy / len(self.scales)

    def compute_kernel_expv(self, x_samples, y_samples):
        exp_k = 0.0
        for x_i, y_i in zip(x_samples, y_samples):
            exp_k += self.compute_kernel(x_i, y_i)

        return exp_k / self.n_shots

    def compute_loss(self, weights):
        exp_k_p_p = self.compute_kernel_expv(
            self.get_circuit_samples(weights), self.get_circuit_samples(weights)
        )
        exp_k_p_pi = self.compute_kernel_expv(
            self.get_circuit_samples(weights), self.get_target_samples()
        )
        exp_k_pi_pi = self.compute_kernel_expv(
            self.get_target_samples(), self.get_target_samples()
        )

        return exp_k_p_p - 2 * exp_k_p_pi + exp_k_pi_pi

    def compute_partial_weight(self, weights, index):
        theta_plus = weights.copy()
        theta_plus.at[index].add(jnp.pi / 2)

        theta_minus = weights.copy()
        theta_minus.at[index].subtract(jnp.pi / 2)

        # p_theta_+, p_theta
        exp_k_p_plus_p = self.compute_kernel_expv(
            self.get_circuit_samples(theta_plus), self.get_circuit_samples(weights)
        )

        # p_theta_-, p_theta
        exp_k_p_minus_p = self.compute_kernel_expv(
            self.get_circuit_samples(theta_minus), self.get_circuit_samples(weights)
        )

        # p_theta_+, pi
        exp_k_p_plus_pi = self.compute_kernel_expv(
            self.get_circuit_samples(theta_plus), self.get_target_samples()
        )

        # p_theta_-, pi
        exp_k_p_minus_pi = self.compute_kernel_expv(
            self.get_circuit_samples(theta_minus), self.get_target_samples()
        )

        return exp_k_p_plus_p - exp_k_p_minus_p - exp_k_p_plus_pi + exp_k_p_minus_pi

    def compute_gradient(self, weights):
        grad = jnp.empty_like(weights)

        for idx in np.ndindex(weights):
            grad.at[idx].set(self.compute_partial_weight(weights, idx))

        return grad


def main():
    pass


if __name__ == "__main__":
    main()
