import time
import numpy as np
import math
import game


class Channel:

    """
    def __init__(self, dino_x, dino_y, bird_x, bird_y, cactus_x, score, hasGameEnded):
        self.dino_x = dino_x
        self.dino_y = dino_y
        self.bird_x = bird_x
        self.bird_y = bird_y
        self.cactus_x = cactus_x
        self.score = score
        self.hasGameEnded = False
    """

    def __init__(self, dino, birds, cacti, score, hasGameEnded):
        self.dino = dino
        self.birds = birds
        self.cacti = cacti
        self.score = score
        self.hasGameEnded = False
        self.successfulJumps = 0

    """
    def print(self):
        print(f"dino_x: {self.dino_x}")
        print(f"dino_y: {self.dino_y}")
        print(f"bird_x: {self.bird_x}")
        print(f"bird_y: {self.bird_y}")
        print(f"cactus_x: {self.cactus_x}")
        print(f"score: {self.score}")
        print(f"hasGameEnded: {self.hasGameEnded}")
    """

    def getEnvironment(self):
        cacti_dist = []
        birds_dist = []
        for cactus in list(self.cacti):
            if cactus.getX() < self.dino.getX():
                self.cacti.remove(cactus)
                self.successfulJumps += 1
            else:
                dist = math.sqrt(
                    math.pow(cactus.getX() - self.dino.getX(), 2) + math.pow(cactus.getY() - self.dino.getY(), 2))
                cacti_dist.append(dist)

        for bird in list(self.birds):
            if bird.getX() < self.dino.getX():
                self.birds.remove(bird)
                self.successfulJumps += 1
            else:
                dist = math.sqrt(
                    math.pow(bird.getX() - self.dino.getX(), 2) + math.pow(bird.getY() - self.dino.getY(), 2))
                birds_dist.append(dist)

        nearest_cactus_dist = min(cacti_dist) if cacti_dist else 1000
        nearest_bird_dist = min(birds_dist) if birds_dist else 1000

        return {
            "agent": np.array([self.dino.getX(), self.dino.getY()], dtype=np.int64),
            "bird": np.array([nearest_bird_dist], dtype=np.int64),
            "cactus": np.array([nearest_cactus_dist], dtype=np.int64)
        }

    # (!) more direct information for AI to learn makes it quicker to learn
    # learning from location of two items vs distance of avoiding object
