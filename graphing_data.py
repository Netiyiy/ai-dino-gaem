from datetime import datetime

cactus_y = [0]
cactus_time_x = []
speed_y = [4]
speed_time_x = []

bird_y = [0]
bird_time_x = []

cactus_x_values = []

maximum = 0
minimum = 999999999


def addBird():
    bird_y.append(bird_y[-1] + 1)
    current_time = datetime.now()
    bird_time_x.append(current_time)


def addCactus(x):
    cactus_y.append(cactus_y[-1] + 1)
    current_time = datetime.now()
    cactus_time_x.append(current_time)
    cactus_x_values.append(x)


def addSpeed(speed):
    speed_y.append(speed)
    current_time = datetime.now()
    speed_time_x.append(current_time)