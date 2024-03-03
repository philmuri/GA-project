import pygame as pg
import numpy as np
from src.common.settings import KEY_COLOR, KEY_SIZE, HEIGHT, WIDTH, OBSTACLE_SPEED
from src.common.obstacle import Obstacle


class Key():
    def __init__(self) -> None:
        self.size = KEY_SIZE
        # key spawns randomly in the inner 33% part of screen dimensions
        self.x = np.random.randint(WIDTH // 3, 2 * WIDTH // 3)
        self.y = np.random.randint(HEIGHT // 3, 2 * HEIGHT // 3)
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
        else:
            self.x -= OBSTACLE_SPEED
