import os
import pygame
import random
import math as m
import numpy as np
from pygame import *

__author__ = "Shivam Shekhar"

### VARIABLES ###

pygame.mixer.pre_init(44100, -16, 2, 2048)  # fix audio delay
pygame.init()

scr_size = (width, height) = (600, 150)
FPS = 60
gravity = 0.6

black = (0, 0, 0)
white = (255, 255, 255)
background_col = (235, 235, 235)

screen = pygame.display.set_mode(scr_size)
clock = pygame.time.Clock()
pygame.display.set_caption("T-Rex Rush")

jump_sound = pygame.mixer.Sound('sprites/jump.wav')
die_sound = pygame.mixer.Sound('sprites/die.wav')
checkPoint_sound = pygame.mixer.Sound('sprites/checkPoint.wav')

PARENT = None

playerInstances = []


class Player:

    def __init__(self):

        self.dino = Dino(self, 44, 47)
        self.highestScore = 0

        # OBSERVATIONS
        self.successfulJumps = 0

        # ACTIONS
        self.KEY_SPACE = False
        self.KEY_DOWN = False
        self.KEY_UP = False

    def resetPlayer(self):

        self.successfulJumps = 0
        self.KEY_SPACE = False
        self.KEY_DOWN = False
        self.KEY_UP = False
        self.dino = Dino(self, 44, 47)

    def getDino(self):
        return self.dino

    def getObservations(self):
        def process_obstacles(obstacles):
            distances = []
            for obstacle in list(obstacles):
                if obstacle.getX() < self.dino.getX():
                    obstacles.remove(obstacle)
                    self.successfulJumps += 1
                else:
                    dist = m.sqrt(
                        m.pow(obstacle.getX() - self.dino.getX(), 2) + m.pow(obstacle.getY() - self.dino.getY(),
                                                                             2))
                    distances.append(dist)
            if len(distances) >= 2:
                distances.sort()
                nearest_dist = distances[0]
                next_nearest_dist = distances[1]
                distance_between_obstacles = next_nearest_dist - nearest_dist
            elif len(distances) == 1:
                nearest_dist = distances[0]
                distance_between_obstacles = -1
            else:
                nearest_dist = -1
                distance_between_obstacles = -1

            return nearest_dist, distance_between_obstacles

        def process_nearest_bird(birds):
            nearest_b = None
            nearest_dist = 1000
            for b in list(birds):
                if (b.getX() + 50) < self.dino.getX():
                    birds.remove(b)
                    if b.getY() != 70:
                        self.successfulJumps += 1
                else:
                    dist = abs(b.getX() - self.dino.getX())
                    #dist = m.sqrt(m.pow(b.getX() - self.dino.getX(), 2) + m.pow(b.getY() - self.dino.getY(), 2))
                    if dist < nearest_dist:
                        nearest_b = b

                        # reward if crouched:
                        if dist < 100:
                            if nearest_b.getY() == 70:
                                if self.dino.isDucking and not self.dino.isJumping:
                                    self.successfulJumps += 10




            return nearest_b

        global PARENT
        nearest_bird_y = -1
        nearest_bird_dist = -1
        nearest_cactus_dist = -1
        distance_between_obstacles = -1
        if PARENT is not None:
            cacti = PARENT.getCacti()
            birds = PARENT.getBirds()
            nearest_cactus_dist, _ = process_obstacles(cacti)
            nearest_bird_dist, _ = process_obstacles(birds)
            nearest_bird = process_nearest_bird(birds)
            if nearest_bird is not None:
                nearest_bird_y = nearest_bird.getY()
            _, distance_between_obstacles = process_obstacles(cacti + birds)

        return {
            "agent": np.array([self.dino.getX(), self.dino.getY()], dtype=np.int64),
            "nearest_cactus_dist": np.array([nearest_cactus_dist], dtype=np.int64),
            "nearest_bird_dist": np.array([nearest_bird_dist], dtype=np.int64),
            "nearest_bird_Y": np.array([nearest_bird_y], dtype=np.int64),
            "distance_between_obstacles": np.array([distance_between_obstacles], dtype=np.int64)
        }
    def sendActions(self, KEY_SPACE, KEY_DOWN, KEY_UP):
        self.KEY_SPACE = KEY_SPACE
        self.KEY_DOWN = KEY_DOWN
        self.KEY_UP = KEY_UP

    def getActions(self):
        return [self.KEY_SPACE, self.KEY_DOWN, self.KEY_UP]


def createPlayer():
    player = Player()
    playerInstances.append(player)
    return player


def getSurvivingPlayer():
    result = None
    for player in playerInstances:
        if not player.getDino().isDead:
            result = player
    return result


class Game:
    global playerInstances

    def __init__(self):

        global PARENT
        PARENT = self

        # observations
        self.birdObjects = []
        self.cactusObjects = []
        self.hasGameEnded = False

        self.high_score = 0
        self.currentScore = 0
        self.gameSpeed = 4
        self.new_ground = Ground(-1 * self.gameSpeed)
        self.scb = Scoreboard()
        self.high_sc = Scoreboard(width * 0.78)
        self.counter = 0

        self.cacti = pygame.sprite.Group()
        self.pteras = pygame.sprite.Group()
        self.clouds = pygame.sprite.Group()
        self.last_obstacle = pygame.sprite.Group()

        Cactus.containers = self.cacti
        Ptera.containers = self.pteras
        Cloud.containers = self.clouds

        self.retbutton_image, self.retbutton_rect = load_image('replay_button.png', 35, 31, -1)
        self.gameover_image, self.gameover_rect = load_image('game_over.png', 190, 11, -1)

        self.temp_images, self.temp_rect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        self.HI_image = pygame.Surface((22, int(11 * 6 / 5)))
        self.HI_rect = self.HI_image.get_rect()
        self.HI_image.fill(background_col)
        self.HI_image.blit(self.temp_images[10], self.temp_rect)
        self.temp_rect.left += self.temp_rect.width
        self.HI_image.blit(self.temp_images[11], self.temp_rect)
        self.HI_rect.top = height * 0.1
        self.HI_rect.left = width * 0.73

    def getCacti(self):
        return self.cactusObjects

    def getBirds(self):
        return self.birdObjects

    def reset(self):
        self.__init__()
        for player in playerInstances:
            player.resetPlayer()

    def play(self):

        if pygame.display.get_surface() is None:
            print("Couldn't load display surface")
        else:

            for player in playerInstances:

                playerDino = player.getDino()
                actions_array = player.getActions()

                KEY_SPACE = actions_array[0]
                KEY_DOWN = actions_array[1]
                KEY_UP = actions_array[2]

                if KEY_SPACE:
                    if playerDino.rect.bottom == int(0.98 * height):
                        playerDino.isJumping = True
                        if pygame.mixer.get_init() is not None:
                            pass
                            # jump_sound.play()
                        playerDino.movement[1] = -1 * playerDino.jumpSpeed

                if KEY_DOWN:
                    if not (playerDino.isJumping and playerDino.isDead):
                        playerDino.isDucking = True

                if KEY_UP:
                    playerDino.isDucking = False

                for c in self.cacti:
                    c.movement[0] = -1 * self.gameSpeed
                    if pygame.sprite.collide_mask(playerDino, c):
                        playerDino.isDead = True
                        if pygame.mixer.get_init() is not None:
                            pass
                            # die_sound.play()

                for p in self.pteras:
                    p.movement[0] = -1 * self.gameSpeed
                    if pygame.sprite.collide_mask(playerDino, p):
                        playerDino.isDead = True
                        if pygame.mixer.get_init() is not None:
                            pass
                            # die_sound.play()

            if len(self.cacti) < 2:
                if len(self.cacti) == 0:
                    self.last_obstacle.empty()
                    self.last_obstacle.add(Cactus(self, self.gameSpeed, 40, 40))
                else:
                    for l in self.last_obstacle:
                        if l.rect.right < width * 0.7 and random.randrange(0, 50) == 10:
                            self.last_obstacle.empty()
                            self.last_obstacle.add(Cactus(self, self.gameSpeed, 40, 40))

            if len(self.pteras) == 0 and random.randrange(0, 20) == 10: #and self.counter > 500: # (STOP 200)
                for l in self.last_obstacle:
                    if l.rect.right < width * 0.8:
                        self.last_obstacle.empty()
                        self.last_obstacle.add(Ptera(self, self.gameSpeed, 46, 40))

            if len(self.clouds) < 5 and random.randrange(0, 300) == 10:
                Cloud(width, random.randrange(height / 5, height / 2))

            for player in playerInstances:
                playerDino = player.getDino()
                playerDino.update()

            self.cacti.update()
            self.pteras.update()
            self.clouds.update()
            self.new_ground.update()
            self.high_sc.update(self.high_score)
            lastPlayer = getSurvivingPlayer()
            if lastPlayer is not None:
                score = lastPlayer.getDino().score
                self.scb.update(score)
                self.currentScore = score
                if score % 100 == 0 and score != 0:
                    if pygame.mixer.get_init() is not None:
                        # checkPoint_sound.play()
                        pass

            if pygame.display.get_surface() is not None:
                screen.fill(background_col)
                self.new_ground.draw()
                self.clouds.draw(screen)
                self.scb.draw()
                if self.high_score != 0:
                    self.high_sc.draw()
                    screen.blit(self.HI_image, self.HI_rect)
                self.cacti.draw(screen)
                self.pteras.draw(screen)
                for player in playerInstances:
                    if not player.getDino().isDead:
                        player.getDino().draw()

                pygame.display.update()
                clock.tick(FPS)

                lastPlayer = getSurvivingPlayer()
                if lastPlayer is None:
                    self.hasGameEnded = True
                    if self.currentScore > self.high_score:
                        self.high_score = self.currentScore

                """
                if self.counter % 700 == 699:
                    self.new_ground.speed -= 1
                    self.gamespeed += 1
                """

                self.counter = (self.counter + 1)


### GAME OBJECTS ###


class Dino:
    def __init__(self, parent, sizex=-1, sizey=-1):

        self.images, self.rect = load_sprite_sheet('dino.png', 5, 1, sizex, sizey, -1)
        self.images1, self.rect1 = load_sprite_sheet('dino_ducking.png', 2, 1, 59, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width / 15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0, 0]
        self.jumpSpeed = 11.5

        self.parent = parent
        self.dino_x = self.rect.left
        self.dino_y = self.rect.top

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        screen.blit(self.image, self.rect)

    def checkbounds(self):
        if self.rect.bottom > int(0.98 * height):
            self.rect.bottom = int(0.98 * height)
            self.isJumping = False

    def update(self):

        self.dino_x = self.rect.left
        self.dino_y = self.rect.top

        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1) % 2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1) % 2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 2 + 2

        if self.isDead:
            self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index) % 2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1
            if self.score > self.parent.highestScore:
                self.parent.highestScore = self.score

        self.counter = (self.counter + 1)

    def getX(self):
        return self.rect.x

    def getY(self):
        return self.rect.y


class Cactus(pygame.sprite.Sprite):
    def __init__(self, parent, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.parent = parent
        self.images, self.rect = load_sprite_sheet('cacti-small.png', 3, 1, sizex, sizey, -1)
        self.rect.bottom = int(0.98 * height)
        self.rect.left = width + self.rect.width
        self.image = self.images[random.randrange(0, 3)]
        self.movement = [-1 * speed, 0]
        parent.cactusObjects.append(self)

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()

    def getX(self):
        return self.rect.x

    def getY(self):
        return self.rect.y


class Ptera(pygame.sprite.Sprite):
    def __init__(self, parent, speed=5, sizex=-1, sizey=-1):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.images, self.rect = load_sprite_sheet('ptera.png', 2, 1, sizex, sizey, -1)
        self.ptera_height = [height * 0.82, height * 0.75, height * 0.60]
        self.rect.centery = self.ptera_height[random.randrange(0, 3)]
        self.rect.left = width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1 * speed, 0]
        self.index = 0
        self.counter = 0
        parent.birdObjects.append(self)

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index + 1) % 2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()

    def getX(self):
        return self.rect.x

    def getY(self):
        return self.rect.y


class Ground:
    def __init__(self, speed=-5):
        self.image, self.rect = load_image('ground.png', -1, -1, -1)
        self.image1, self.rect1 = load_image('ground.png', -1, -1, -1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image, self.rect)
        screen.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right


class Cloud(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image, self.rect = load_image('cloud.png', int(90 * 30 / 42), 30, -1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1 * self.speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


class Scoreboard:
    def __init__(self, x=-1, y=-1):
        self.score = 0
        self.tempimages, self.temprect = load_sprite_sheet('numbers.png', 12, 1, 11, int(11 * 6 / 5), -1)
        self.image = pygame.Surface((55, int(11 * 6 / 5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = width * 0.89
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = height * 0.1
        else:
            self.rect.top = y

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self, score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s], self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


### METHODS ###


def load_image(
        name,
        sizex=-1,
        sizey=-1,
        colorkey=None,
):
    fullname = os.path.join('sprites', name)
    image = pygame.image.load(fullname)
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))

    return (image, image.get_rect())


def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scalex=-1,
        scaley=-1,
        colorkey=None,
):
    fullname = os.path.join('sprites', sheetname)
    sheet = pygame.image.load(fullname)
    sheet = sheet.convert()

    sheet_rect = sheet.get_rect()

    sprites = []

    sizex = sheet_rect.width / nx
    sizey = sheet_rect.height / ny

    for i in range(0, ny):
        for j in range(0, nx):
            rect = pygame.Rect((j * sizex, i * sizey, sizex, sizey))
            image = pygame.Surface(rect.size)
            image = image.convert()
            image.blit(sheet, (0, 0), rect)

            if colorkey is not None:
                if colorkey == -1:
                    colorkey = image.get_at((0, 0))
                image.set_colorkey(colorkey, RLEACCEL)

            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image, (scalex, scaley))

            sprites.append(image)

    sprite_rect = sprites[0].get_rect()

    return sprites, sprite_rect


def disp_gameOver_msg(retbutton_image, gameover_image):
    retbutton_rect = retbutton_image.get_rect()
    retbutton_rect.centerx = width / 2
    retbutton_rect.top = height * 0.52

    gameover_rect = gameover_image.get_rect()
    gameover_rect.centerx = width / 2
    gameover_rect.centery = height * 0.35

    screen.blit(retbutton_image, retbutton_rect)
    screen.blit(gameover_image, gameover_rect)


def extractDigits(number):
    if number > -1:
        digits = []
        i = 0
        while number / 10 != 0:
            digits.append(number % 10)
            number = int(number / 10)

        digits.append(number % 10)
        for i in range(len(digits), 5):
            digits.append(0)
        digits.reverse()
        return digits
