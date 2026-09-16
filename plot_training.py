import csv

import matplotlib.pyplot as plt


HISTORY_FILE = (
    "training_history_continuous.csv"
)


generations = []
best_fitness = []
average_fitness = []
best_laps = []
average_speed = []


with open(
    HISTORY_FILE,
    "r"
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        generations.append(
            int(
                row["generation"]
            )
        )

        best_fitness.append(
            float(
                row["best_fitness"]
            )
        )

        average_fitness.append(
            float(
                row["average_fitness"]
            )
        )

        best_laps.append(
            int(
                row["best_laps"]
            )
        )

        average_speed.append(
            float(
                row["average_speed"]
            )
        )


# ==================================================
# FITNESS
# ==================================================

plt.figure()

plt.plot(
    generations,
    best_fitness,
    label="Best fitness"
)

plt.plot(
    generations,
    average_fitness,
    label="Average fitness"
)

plt.xlabel(
    "Generation"
)

plt.ylabel(
    "Fitness"
)

plt.title(
    "Continuous Control Training"
)

plt.legend()

plt.grid()

plt.show()


# ==================================================
# LAPS
# ==================================================

plt.figure()

plt.plot(
    generations,
    best_laps
)

plt.xlabel(
    "Generation"
)

plt.ylabel(
    "Best laps"
)

plt.title(
    "Best Laps per Generation"
)

plt.grid()

plt.show()


# ==================================================
# SPEED
# ==================================================

plt.figure()

plt.plot(
    generations,
    average_speed
)

plt.xlabel(
    "Generation"
)

plt.ylabel(
    "Normalized average speed"
)

plt.title(
    "Champion Average Speed"
)

plt.grid()

plt.show()