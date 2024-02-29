import numpy as np
from constants import OBSTACLE_COLOR, OBSTACLE_SPEED, OBSTACLE_WIDTH, OBSTACLE_HEIGHT_MIN, OBSTACLE_HEIGHT_MAX, HEIGHT, WIDTH, BASE_HEIGHT
import pygame as pg


class Obstacle():
    def __init__(self) -> None:
        self.width = OBSTACLE_WIDTH
        self.height = np.random.randint(
            OBSTACLE_HEIGHT_MIN, OBSTACLE_HEIGHT_MAX)
        self.x = WIDTH
        self.y = HEIGHT - BASE_HEIGHT - self.height

    def draw(self, screen) -> None:
        pg.draw.rect(screen, OBSTACLE_COLOR,
                     (self.x, self.y, self.width, self.height))

    def update(self) -> None:
        if self.x + self.width <= 0:
            self.__init__()  # bad practice, but works
        self.x -= OBSTACLE_SPEED
