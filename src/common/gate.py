import pygame as pg
from src.common.settings import GATE_CLOSED_COLOR, OBSTACLE_SPEED, OBSTACLE_WIDTH, HEIGHT, WIDTH, BASE_HEIGHT
from src.common.obstacle import Obstacle


class Gate():
    def __init__(self, obstacle: Obstacle) -> None:
        self.width = OBSTACLE_WIDTH
        self.x = WIDTH
        self.is_open = False
        if obstacle.category == 'bottom':
            self.y = BASE_HEIGHT
            self.height = obstacle.y - BASE_HEIGHT
        else:
            self.y = obstacle.y
            self.height = HEIGHT - BASE_HEIGHT - obstacle.y

    def draw(self, screen) -> None:
        if not self.is_open:
            pg.draw.rect(screen, GATE_CLOSED_COLOR,
                         (self.x, self.y, self.width, self.height))
        else:
            pass

    def update(self, obstacle: Obstacle) -> None:
        if self.is_outside():
            self.__init__(obstacle)
        self.x -= OBSTACLE_SPEED

    def is_outside(self) -> bool:
        if self.x + self.width <= 0:
            return True
        else:
            return False


"""
TBD:
- Add a "lock" icon to the gate center, which is an "open lock" when unlocked
- Consider inheritance
"""
