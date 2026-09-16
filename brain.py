import numpy as np


class Brain:
    def __init__(self):
        # Inputs:
        # 5 sensors + speed = 6
        self.weights1 = np.random.randn(6, 8)
        self.bias1 = np.random.randn(8)

        # Outputs:
        # 0 = steering
        # 1 = throttle
        self.weights2 = np.random.randn(8, 2)
        self.bias2 = np.random.randn(2)

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

        raw_output = (
            hidden @ self.weights2
            + self.bias2
        )

        output = np.tanh(
            raw_output
        )

        # Steering stays in [-1, 1]
        steering = float(
            output[0]
        )

        # Convert throttle:
        # [-1, 1] -> [0, 1]
        throttle = float(
            (output[1] + 1.0) / 2.0
        )

        return steering, throttle

    # ==================================================
    # COPY
    # ==================================================

    def copy(self):
        brain = Brain()

        brain.weights1 = (
            self.weights1.copy()
        )

        brain.bias1 = (
            self.bias1.copy()
        )

        brain.weights2 = (
            self.weights2.copy()
        )

        brain.bias2 = (
            self.bias2.copy()
        )

        return brain

    # ==================================================
    # MUTATION
    # ==================================================

    def mutate(self, rate=0.05):
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
        data = np.load(
            filename
        )

        # Basic compatibility check
        if data["weights2"].shape != (8, 2):
            raise ValueError(
                "Saved brain has an incompatible "
                "output layer. Continuous-control "
                "brain requires shape (8, 2)."
            )

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