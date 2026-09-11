import numpy as np


class Brain:
    def __init__(self):
        # 6 inputs:
        # 5 sensors + speed
        self.weights1 = np.random.randn(6, 8)
        self.bias1 = np.random.randn(8)

        # 3 outputs:
        # 0 = left
        # 1 = straight
        # 2 = right
        self.weights2 = np.random.randn(8, 3)
        self.bias2 = np.random.randn(3)

    # ==================================================
    # FORWARD PASS
    # ==================================================

    def predict(self, inputs):
        inputs = np.array(
            inputs,
            dtype=float
        )

        hidden = np.tanh(
            inputs @ self.weights1
            + self.bias1
        )

        output = (
            hidden @ self.weights2
            + self.bias2
        )

        return int(
            np.argmax(output)
        )

    # ==================================================
    # COPY
    # ==================================================

    def copy(self):
        new_brain = Brain()

        new_brain.weights1 = (
            self.weights1.copy()
        )

        new_brain.bias1 = (
            self.bias1.copy()
        )

        new_brain.weights2 = (
            self.weights2.copy()
        )

        new_brain.bias2 = (
            self.bias2.copy()
        )

        return new_brain

    # ==================================================
    # MUTATION
    # ==================================================

    def mutate(self, rate=0.15):
        self.weights1 += (
            np.random.randn(
                *self.weights1.shape
            )
            * rate
        )

        self.bias1 += (
            np.random.randn(
                *self.bias1.shape
            )
            * rate
        )

        self.weights2 += (
            np.random.randn(
                *self.weights2.shape
            )
            * rate
        )

        self.bias2 += (
            np.random.randn(
                *self.bias2.shape
            )
            * rate
        )

    # ==================================================
    # CROSSOVER
    # ==================================================

    @staticmethod
    def crossover(
        parent_a,
        parent_b
    ):
        child = Brain()

        mask = (
            np.random.rand(
                *parent_a.weights1.shape
            )
            < 0.5
        )

        child.weights1 = np.where(
            mask,
            parent_a.weights1,
            parent_b.weights1
        )

        mask = (
            np.random.rand(
                *parent_a.bias1.shape
            )
            < 0.5
        )

        child.bias1 = np.where(
            mask,
            parent_a.bias1,
            parent_b.bias1
        )

        mask = (
            np.random.rand(
                *parent_a.weights2.shape
            )
            < 0.5
        )

        child.weights2 = np.where(
            mask,
            parent_a.weights2,
            parent_b.weights2
        )

        mask = (
            np.random.rand(
                *parent_a.bias2.shape
            )
            < 0.5
        )

        child.bias2 = np.where(
            mask,
            parent_a.bias2,
            parent_b.bias2
        )

        return child

    # ==================================================
    # SAVE
    # ==================================================

    def save(self, filename):
        np.savez(
            filename,
            weights1=self.weights1,
            bias1=self.bias1,
            weights2=self.weights2,
            bias2=self.bias2
        )

    # ==================================================
    # LOAD
    # ==================================================

    @staticmethod
    def load(filename):
        data = np.load(filename)

        brain = Brain()

        brain.weights1 = (
            data["weights1"].copy()
        )

        brain.bias1 = (
            data["bias1"].copy()
        )

        brain.weights2 = (
            data["weights2"].copy()
        )

        brain.bias2 = (
            data["bias2"].copy()
        )

        return brain