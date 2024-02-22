import pygame as pg
import numpy as np
import time
import sys

# Constants
WIDTH, HEIGHT = 800, 600
BASE_HEIGHT = 100
OBSTACLE_WIDTH = 25
OBSTACLE_HEIGHT = 200
BG_COLOR = (200, 200, 200)
BASE_COLOR = (0, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)

GAME_SPEED = 5
GRAVITY = 1
JUMP_FORCE = -10


class Player:
    def __init__(self): # Initialize character
        self.radius = 20
        self.x = 40
        self.y = HEIGHT - BASE_HEIGHT - self.radius
        self.vy = 0
        self.is_jumping = False
        self.color = (0, 0, 255)
    
    def draw(self):
        pg.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def update(self):
        self.gravity()
        self.y += self.vy

    def gravity(self):
        if self.y >= HEIGHT - BASE_HEIGHT - self.radius and self.vy >= 0: # if player near (or under) ground while having downward or no velocity, terminate jump and reset player to ground level with zero velocity
            self.y = HEIGHT - BASE_HEIGHT - self.radius
            self.vy = 0
            self.is_jumping = False
        else: # if in air and upward velocity, act with gravity (downward constant acceleration)
            self.vy += GRAVITY   

    def jump(self):
        self.is_jumping = True
        self.vy = JUMP_FORCE 

    def is_colliding(self, obstacle):
        dx = self.x - max(obstacle.x, min(self.x, obstacle.x + obstacle.width)) # a smart way to get the nearest x-coordinate of the obstacle to the player's center
        dy = self.y - max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
        dist = dx**2 + dy**2 # squared distance
        return dist < self.radius**2 # objects colliding if squared distance smaller than radius squared


class Obstacle:
    def __init__(self, height): # obstacle height will be randomized
        self.width = OBSTACLE_WIDTH
        self.height = height
        self.x = WIDTH
        self.y = HEIGHT - BASE_HEIGHT - self.height

    def draw(self):
        pg.draw.rect(screen, OBSTACLE_COLOR, (self.x, self.y, self.width, self.height))



if __name__ == '__main__':
    # Initialize game
    pg.init()
    clock = pg.time.Clock()

    player = Player()
    obstacle = Obstacle(height=OBSTACLE_HEIGHT)

    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Obstacle Jumping")

    # Run game
    game_running = True
    game_paused = False

    while game_running: # handle game events first
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    player.jump()
                elif event.key == pg.K_p:
                    game_paused = not game_paused

        if not game_paused:

            screen.fill(BG_COLOR)
            pg.draw.rect(screen, BASE_COLOR, (0, HEIGHT - BASE_HEIGHT, WIDTH, HEIGHT)) # draw baseplate

            # Update game objects: update player position, update obstacle position
            player.update()
            # TBD: introduce as new obstacle method update(self)
            if obstacle.x < -OBSTACLE_WIDTH:
                obstacle.x = WIDTH # set obstacle back to the right
            obstacle.x -= GAME_SPEED # TBD: replace with obstacle.vx
            
            # Handle collision event
            if player.is_colliding(obstacle=obstacle):
                game_paused = True # TBD: Replace with game reset after small pause (e.g. with time.sleep())

            # Draw objects
            player.draw()
            obstacle.draw()

            pg.display.flip() # update screen display

        clock.tick(60) # frames per second

    pg.quit()
    sys.exit()

    """
    TBD:
    - Add reset game function: Sets the player and object back to the starting position 
    - Add option to have multiple players spawn (for now at same starting position)
    - Implement the genetic algorithm into the game:
        - Think about the parameters in the game that translate into genes for the GA, such as jumping 
        - Think about how to translate player performance into a fitness function (potentially maximizing distance travelled/time alive/obstacles avoided)
    """