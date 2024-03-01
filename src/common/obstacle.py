import numpy as np
from src.common.settings import OBSTACLE_COLOR, OBSTACLE_SPEED, OBSTACLE_WIDTH, OBSTACLE_HEIGHT_MIN, OBSTACLE_HEIGHT_MAX, HEIGHT, WIDTH, BASE_HEIGHT
import pygame as pg
import random


class Obstacle():
    score: int = 0

    def __init__(self) -> None:
        self.category = random.choice(['bottom', 'top'])
        self.width = OBSTACLE_WIDTH
        self.height = np.random.randint(
            OBSTACLE_HEIGHT_MIN, OBSTACLE_HEIGHT_MAX)
        self.x = WIDTH
        if self.category == 'bottom':
            self.y = HEIGHT - BASE_HEIGHT - self.height
        else:
            self.y = BASE_HEIGHT + self.height

    def draw(self, screen) -> None:
        if self.category == 'bottom':
            pg.draw.rect(screen, OBSTACLE_COLOR,
                         (self.x, self.y, self.width, self.height))
        else:
            pg.draw.rect(screen, OBSTACLE_COLOR,
                         (self.x, self.y - self.height, self.width, self.height))

    def update(self) -> None:
        if self.x + self.width <= 0:
            self.score += 1
            self.__init__()  # bad practice, but works
        self.x -= OBSTACLE_SPEED
