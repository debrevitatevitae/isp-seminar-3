import pdb

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


class BarsAndStripes:

    def __init__(self, n: int) -> None:
        self.n_pixels = n
        self.size = n ** 2

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

        self.data = np.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))

        self.bitstrings = []
        self.int_labels = []
        for d in self.data:
            self.bitstrings += ["".join(str(int(i)) for i in d)]
            self.int_labels += [int(self.bitstrings[-1], 2)]


    def plot_sample(self, sample_id: int) -> matplotlib.figure.Figure:
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
                text = plt.text(
                    i,
                    j,
                    sample[j][i],
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=12,
                )

        return plt.gcf()


    def plot_dataset(self) -> matplotlib.figure.Figure:
        n = self.n_pixels

        plt.figure(figsize=(4, 4))
        j = 1
        for i in self.data:
            plt.subplot(4, 4, j)
            j += 1
            plt.imshow(np.reshape(i, (n, n)), cmap="gray", vmin=0, vmax=1)
            plt.xticks([])
            plt.yticks([])

        return plt.gcf()


    def plot_data_dist(self) -> matplotlib.figure.Figure:
        probs = np.zeros(2**self.size)
        probs[self.int_labels] = 1 / len(self.data)

        plt.figure(figsize=(12, 5))
        plt.bar(np.arange(2**self.size), probs, width=2.0, label=r"$\pi(x)$")
        plt.xticks(self.int_labels, self.bitstrings, rotation=80)

        plt.xlabel("Samples")
        plt.ylabel("Prob. Distribution")
        plt.legend(loc="upper right")
        plt.subplots_adjust(bottom=0.3)

        return plt.gcf()



def main():
    pass

if __name__ == "__main__":
    main()
