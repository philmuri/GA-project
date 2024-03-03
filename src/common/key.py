import pygame as pg
import numpy as np
from src.common.settings import KEY_COLOR, KEY_SIZE, PLAYER_START_POS, HEIGHT, WIDTH, BASE_HEIGHT, OBSTACLE_SPEED
from src.common.obstacle import Obstacle


class Key():
    def __init__(self) -> None:
        self.x = np.random.randint(PLAYER_START_POS + 100, WIDTH - 100)
        self.y = np.random.randint(BASE_HEIGHT + 50, HEIGHT - BASE_HEIGHT - 50)
        self.size = KEY_SIZE
        self.is_collected = False

    def draw(self, screen) -> None:
        if not self.is_collected:
            pg.draw.rect(screen, KEY_COLOR,
                         (self.x, self.y, self.size, self.size))
        else:
            pass

    def update(self, obstacle: Obstacle) -> None:
        if obstacle.is_outside():
            self.__init__()
        self.x -= OBSTACLE_SPEED

    def is_outside(self) -> bool:
        if self.x + self.size <= 0:
            return True
        else:
            return False


"""
TBD:
- Key durability? Requires x fraction of current players to pass
"""
