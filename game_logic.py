import pygame as pg
import numpy as np
import time
import sys
import pygad

# Constants
WIDTH, HEIGHT = 800, 600
BASE_HEIGHT = 50

OBSTACLE_WIDTH_MIN = 50
OBSTACLE_WIDTH_MAX = 200
OBSTACLE_HEIGHT_MIN = 50
OBSTACLE_HEIGHT_MAX = 300

PLAYER_RADIUS = 10
POPULATION_SIZE = 10

BG_COLOR = (200, 200, 200)
BASE_COLOR = (0, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)
PLAYER_COLOR = (128, 128, 128)

OBSTACLE_SPEED = 5
GRAVITY = 0.5
JUMP_FORCE = -10
RESET_TIMER = 1 # temporary reset delay after collision (seconds)


class Player:
    def __init__(self, genes):
        self.radius = PLAYER_RADIUS
        self.x = 40
        self.y = HEIGHT - BASE_HEIGHT - self.radius
        self.vy = 0
        self.is_jumping = False
        self.color = PLAYER_COLOR
        self.genes = genes
        """
        {
            'min_dx_obstacle': np.random.randint(50, 200), # minimum player-obstacle distance before jump
            'min_dy_obstacle': np.random.randint(50, 200) # minimum player-obstacle height difference before jump
        }
        """
    
    def draw(self, screen):
        pg.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def update(self, obstacle=None):
        self.gravity()
        self.y += self.vy
        if self.genes and self.should_jump(obstacle):
            self.jump()

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
        dist = dx**2 + dy**2
        return dist < self.radius**2 or self.y <= BASE_HEIGHT # also handles roof collision
    
    def should_jump(self, obstacle): # for GA training only
        dist = obstacle.x - self.x # + when left of obstacle, - right when player center crossing obstacle
        height_diff = self.y - obstacle.y # + when below obstacle, - when player center crossing above obstacle
        if (dist + obstacle.width >= 0 and dist <= self.genes[0]): # player ignores current obstacle distance once the center passes its right side. player jumps before then only if also its distance to obstacle is less than the gene-specified value
            return True
        if (height_diff >= self.genes[1]): # if player is sufficiently below an obstacle, jump
            return True
        return False


class Obstacle:
    def __init__(self): # obstacle height will be randomized
        self.width = np.random.randint(OBSTACLE_WIDTH_MIN, OBSTACLE_WIDTH_MAX)
        self.height = np.random.randint(OBSTACLE_HEIGHT_MIN, OBSTACLE_HEIGHT_MAX)
        self.x = WIDTH
        self.y = HEIGHT - BASE_HEIGHT - self.height

    def draw(self, screen):
        pg.draw.rect(screen, OBSTACLE_COLOR, (self.x, self.y, self.width, self.height))

    def update(self):
        if self.x + self.width <= 0:
                self.__init__() # this might be bad practice
        self.x -= OBSTACLE_SPEED



def run_game_ga():
    # -- Initialize game --
    pg.init() 
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Obstacle Jumping")
    clock = pg.time.Clock()
    
    # -- Initialize players and obstacles --
    # - Populate the game with players
    population = []
    for _ in range(POPULATION_SIZE):
        dist_threshold = np.random.randint(50, 100)
        height_threshold = np.random.randint(50, 100)
        genes = [dist_threshold, height_threshold]
        player = Player(genes=genes)
        population.append(player)
    
    # - Initialize obstacle
    obstacle = Obstacle()


    # -- Run game --
    game_running = True
    game_paused = False
    while game_running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                        game_paused = not game_paused

        if not game_paused:
            screen.fill(BG_COLOR)
            pg.draw.rect(screen, BASE_COLOR, (0, HEIGHT - BASE_HEIGHT, WIDTH, HEIGHT)) # Ground
            pg.draw.rect(screen, BASE_COLOR, (0, 0, WIDTH, BASE_HEIGHT)) # Roof

            # - Update players and obstacle
            for player in population:
                player.update(obstacle)
            obstacle.update()

            for player in population:
                if player.is_colliding(obstacle=obstacle):
                    game_paused = True

            # TBD: Handle removal of collided players and reset game when all players have collided
            for player in population:
                player.draw(screen)
            obstacle.draw(screen)

            pg.display.flip()

        clock.tick(60)

    pg.quit()
    sys.exit()




# STANDARD GAME
def reset_game(player, obstacle):
    player.__init__()
    obstacle.__init__()


def run_game():
    # Initialize game
    pg.init()
    clock = pg.time.Clock()
    player = Player()
    obstacle = Obstacle()
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
            obstacle.update()
            
            # Handle collision event
            if player.is_colliding(obstacle):
                time.sleep(RESET_TIMER)
                reset_game(player, obstacle)


            # Draw objects
            player.draw(screen)
            obstacle.draw(screen)

            pg.display.flip() # update screen display

        clock.tick(60) # frames per second

    pg.quit()
    sys.exit()



if __name__ == '__main__':
    run_game_ga()


"""
TBD:
- Make jump force another gene to be evolved by genetic algorithm
- Put a lower limit on jump rate
- Handle multiple players: Continue game even if one collides, end game only when all collidedd
- Add reset game function: Sets the player and object back to the starting position 
- Add option to have multiple players spawn (for now at same starting position)
- Implement the genetic algorithm into the game: --> Start by implementing from GA from scratch?
    - Think about the parameters in the game that translate into genes for the GA, such as jump timing, jump count, distance to obstacle before jump, height above obstacle before jump, ... 
    - Think about how to translate player performance into a fitness function (potentially maximizing distance travelled/time alive/obstacles avoided)
"""