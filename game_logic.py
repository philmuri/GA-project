import pygame as pg
import numpy as np
import time
import sys
import pygad

# Constants
WIDTH, HEIGHT = 800, 600
BASE_HEIGHT = 100

OBSTACLE_WIDTH_MIN = 25
OBSTACLE_WIDTH_MAX = 100
OBSTACLE_HEIGHT_MIN = 50
OBSTACLE_HEIGHT_MAX = 300

PLAYER_RADIUS = 15

BG_COLOR = (200, 200, 200)
BASE_COLOR = (0, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)
PLAYER_COLOR = (128, 128, 128)

OBSTACLE_SPEED = 5
GRAVITY = 0.5
JUMP_FORCE = -10
RESET_TIMER = 1 # temporary reset delay after collision (seconds)


class Player:
    def __init__(self):
        self.radius = PLAYER_RADIUS
        self.x = 40
        self.y = HEIGHT - BASE_HEIGHT - self.radius
        self.vy = 0
        self.is_jumping = False
        self.color = PLAYER_COLOR
        self.chromosome = [np.random.randint(50, 200), np.random.randint(50, 200)]
        """
        {
            'min_dx_obstacle': np.random.randint(50, 200), # minimum player-obstacle distance before jump
            'min_dy_obstacle': np.random.randint(50, 200) # minimum player-obstacle height difference before jump
        }
        """
    
    def draw(self, screen):
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
        # TBD: potentially add a pause before this function can be called succesfully again, to prevent spam jumping

    def is_colliding(self, obstacle):
        dx = self.x - max(obstacle.x, min(self.x, obstacle.x + obstacle.width)) # a smart way to get the nearest x-coordinate of the obstacle to the player's center
        dy = self.y - max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
        dist = dx**2 + dy**2
        return dist < self.radius**2
        # TBD: Add a roof that can also be collided with. Only need to track y-distance for this
    
    def update_GA(self, obstacle):
        self.gravity()
        self.y += self.vy
        if any(self.chromosome) and self.should_jump(obstacle): # for GA training only (if None, this is ignored)
                self.jump()
    
    def should_jump(self, obstacle): # for GA training only
        dx = self.x - max(obstacle.x, min(self.x, obstacle.x + obstacle.width))
        dy = self.y - max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
        if dx == self.chromosome[0] or dy == self.chromosome[1]:
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


# GENETIC ALGORITHM GAME
def get_player_fitness(ga_instance, solution, solution_idx):
    player = Player()
    obstacle = Obstacle()
    player.chromosome = solution

    while not player.is_colliding(obstacle):
        player.update_GA(obstacle)
        obstacle.update()

    fitness_value = player.x # for now, the only measure of fitness is the distance the player has traveled
    return fitness_value


def reset_game_ga(player, obstacle):
    player.x = 40
    player.y = HEIGHT - BASE_HEIGHT - player.radius
    player.vy = 0
    obstacle.__init__()


def run_game_ga():
    # Initialize game
    pg.init()
    clock = pg.time.Clock()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Obstacle Jumping")

    # PYGAD: Run simulations to evolve players
    num_generations = 100
    player_population_size = 10
    num_parents_mating = 5

    ga_instance = pygad.GA(num_generations=num_generations,
                           fitness_func=get_player_fitness,
                           sol_per_pop=player_population_size,
                           num_parents_mating=num_parents_mating,
                           num_genes=len(Player().chromosome),
                           mutation_num_genes=1
                           )

    ga_instance.run() # run GA to evolve player through generations

    best_solution, best_solution_fitness, solution_idx = ga_instance.best_solution()
    best_player = Player()
    obstacle = Obstacle()
    best_player.chromosome = best_solution

    # Run actual game
    game_running = True
    game_paused = False
    while game_running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False

        if not game_paused:
            screen.fill(BG_COLOR)
            pg.draw.rect(screen, BASE_COLOR, (0, HEIGHT - BASE_HEIGHT, WIDTH, HEIGHT))

            best_player.update_GA(obstacle)
            obstacle.update()

            if best_player.is_colliding(obstacle=obstacle):
                reset_game_ga(best_player, obstacle)

            best_player.draw(screen)
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
- Add reset game function: Sets the player and object back to the starting position 
- Add option to have multiple players spawn (for now at same starting position)
- Implement the genetic algorithm into the game:
    - Think about the parameters in the game that translate into genes for the GA, such as jump timing, jump count, distance to obstacle before jump, height above obstacle before jump, ... 
    - Think about how to translate player performance into a fitness function (potentially maximizing distance travelled/time alive/obstacles avoided)
"""