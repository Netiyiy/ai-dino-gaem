from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import graphing_data
import wrapper


def plot(x, y, label_x, label_y, title):
    plt.plot(x, y)

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

    plt.show()


def plotFormatted(x, y, label_y, title):
    formatted_x = [(t - x[0]).total_seconds() for t in x]
    plot(formatted_x, y[1:], "Time (sec)", label_y, title)


def plotFormattedDerivative(x, y, label_y, title):
    formatted_x = [(t - x[0]).total_seconds() for t in x]

    formatted_x = np.array(formatted_x)
    y = np.array(y)

    dy_dx = np.diff(y[1:]) / np.diff(formatted_x)

    # To match the length of x and y, create a new x array for the derivative
    x_derivative = (formatted_x[:-1] + formatted_x[1:]) / 2

    plot(x_derivative, dy_dx, "Time (sec)", label_y, title)


wrapper.runHardCodedTillDeath()

graphing_data.addSpeed(4)

plotFormatted(graphing_data.cactus_time_x, graphing_data.cactus_y, "Total Cactus Seen", "Total Cactus Seen vs Time")
plotFormattedDerivative(graphing_data.cactus_time_x, graphing_data.cactus_y, "Cactus Spawn Rate",
                        "Cactus Spawn Rate vs Time")

plotFormatted(graphing_data.bird_time_x, graphing_data.bird_y, "Total Bird Seen", "Total Bird Seen vs Time")
plotFormattedDerivative(graphing_data.bird_time_x, graphing_data.bird_y, "Bird Spawn Rate", "Bird Spawn Rate vs Time")

plotFormatted(graphing_data.speed_time_x, graphing_data.speed_y, "Game Speed (pixels per frame)", "Game Speed vs Time")

print("Maximum Cactus Distance: " + str(graphing_data.maximum) + " pixels")
print("Minimum Cactus Distance: " + str(graphing_data.minimum) + " pixels")
