import pygame as pg
import numpy as np
import time
import sys
import pygad

# Constants: Game
WIDTH, HEIGHT = 800, 600
BASE_HEIGHT = 50

OBSTACLE_WIDTH_MIN = 50
OBSTACLE_WIDTH_MAX = 200
OBSTACLE_HEIGHT_MIN = 50
OBSTACLE_HEIGHT_MAX = 300

PLAYER_START_POSITION = 80
PLAYER_RADIUS = 15
POPULATION_SIZE = 3

BG_COLOR = (200, 200, 200)
BASE_COLOR = (0, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)
PLAYER_COLOR = (128, 128, 128)
PLAYER_DEATH_COLOR = (255, 0, 0)

OBSTACLE_SPEED = 5
GRAVITY = 0.5
JUMP_FORCE = -10
RESET_TIMER = 1 # temporary reset delay after collision (seconds)

# Constants: Genetic Algorithm
MUTATION_RATE = 0.5


class Player:
    def __init__(self, genes=None):
        self.radius = PLAYER_RADIUS
        self.x = PLAYER_START_POSITION
        self.y = HEIGHT - BASE_HEIGHT - self.radius
        self.vy = 0
        self.is_jumping = False
        self.jump_cooldown = 0.5
        self.last_jump_time = 0
        self.color = PLAYER_COLOR
        self.genes = genes
        self.time_alive = 0
        self.init_time = time.time()
        self.is_animating = False
        self.animation_start_time = 0
        self.is_dead = False
    

    def draw(self, screen, color):
        """
        Draw the player in its current position or according to an animation.
        """
        if self.is_animating:
            self.animation(screen) # separate draw animation
        else:
            pg.draw.circle(screen, color, (self.x, self.y), self.radius)

    def update(self, obstacle=None):
        """
        Handle player motion by updating the player y-position according to simple equation of motion.
        Also handles genetic algorithm player motion by calling jump() when should_jump() is True.
        """
        if self.is_animating:
            pass # disable physics on animating player
        else:
            self.gravity()
            self.y += self.vy
            if self.genes and self.should_jump(obstacle):
                current_time = time.time()
                if current_time - self.last_jump_time >= self.jump_cooldown or self.last_jump_time == 0:
                    self.jump()
                    self.last_jump_time = time.time()


    def gravity(self):
        """
        Simulate game physics as a constant gravitational downforce and player and base interaction.
        """
        if self.y >= HEIGHT - BASE_HEIGHT - self.radius and self.vy >= 0: # if player near (or under) ground while having downward or no velocity, terminate jump and reset player to ground level with zero velocity
            self.y = HEIGHT - BASE_HEIGHT - self.radius
            self.vy = 0
            self.is_jumping = False
        else: # if in air and upward velocity, act with gravity (downward constant acceleration)
            self.vy += GRAVITY   

    def jump(self):
        """
        Jump event for player.
        TBD: Delay, possibly using ratelimit library
        """
        self.is_jumping = True
        self.vy = JUMP_FORCE

    def is_colliding(self, obstacle):
        """
        Handle collision events by evaluating player-obstacle Euclidean distance and player-roof distance.
        Returns True if colliding.
        """
        dx = self.x - max(obstacle.x, min(self.x, obstacle.x + obstacle.width)) # a smart way to get the nearest x-coordinate of the obstacle to the player's center
        dy = self.y - max(obstacle.y, min(self.y, obstacle.y + obstacle.height))
        dist = dx**2 + dy**2
        return dist < self.radius**2 or self.y <= BASE_HEIGHT # also handles roof collision
    
    def should_jump(self, obstacle): # for GA training only
        """
        Jump logic for genetic algorithm training, based on player proximity to obstacle in x and y direction (separately).
        This introduces functionality to the genes in the player chromosome and is the basis for the evolution process.
        (NOTE: for genetic algorithm part only)
        """
        dist = obstacle.x - self.x # + when left of obstacle, - right when player center crossing obstacle
        height_diff = self.y - obstacle.y # + when below obstacle, - when player center crossing above obstacle
        if (dist + obstacle.width >= 0 and dist <= self.genes[0]): # player ignores current obstacle distance once the center passes its right side. player jumps before then only if also its distance to obstacle is less than the gene-specified value
            return True
        if (height_diff >= self.genes[1]): # if player is sufficiently below an obstacle, jump
            return True
        return False
    
    def kill(self): # if kill is called, player object enters is_animating state and will cease to update() while draw() has a custom behavior (in this case moving with obstacle speed)
        """
        Kill a player by toggling the is_animating flag to True, which subsequently disables its physics in the update() method and enables a death animation in draw().
        Also sets the animation_start_time to be used by the animation() method for animation duration measurement.
        """
        self.time_alive = round(time.time() - self.init_time, 3)
        self.is_dead = True
        self.is_animating = True
        self.animation_start_time = time.time()

    def animation(self, screen):
        """
        Draw animation on player death. Called by draw() if is_animating is True.
        """
        duration = (self.x + self.radius) / OBSTACLE_SPEED
        if time.time() - self.animation_start_time >= duration:
            self.is_animating = False
        else:
            self.x -= OBSTACLE_SPEED
            pg.draw.circle(screen, PLAYER_DEATH_COLOR, (self.x, self.y), self.radius)


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


def reset_game_ga(population, obstacle):
    obstacle.__init__()
    for player in population:
        player.__init__() # DO NOT initialize genes, eevrything else yes. Ask ChatGPT how to re-initialize a charcaters attributes except 1


def run_game_ga():
    # -- Initialize game --
    pg.init() 
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Obstacle Jumping")
    clock = pg.time.Clock()
    
    # -- Initialize players and obstacles --
    # - Populate the game with players and initialize their genes
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
            # - Draw background, platform and roof
            screen.fill(BG_COLOR)
            pg.draw.rect(screen, BASE_COLOR, (0, HEIGHT - BASE_HEIGHT, WIDTH, HEIGHT)) # Ground
            pg.draw.rect(screen, BASE_COLOR, (0, 0, WIDTH, BASE_HEIGHT)) # Roof
            # - Update players and obstacle
            for player in population:
                player.update(obstacle)
            obstacle.update()
            # - Handle player-obstacle collision events
            for player in population:
                if not player.is_dead and player.is_colliding(obstacle=obstacle):
                    player.kill()
                    
            # - Redraw players and obstacle
            for player in population:
                player.draw(screen, player.color)
            obstacle.draw(screen)
            # - Handle case when all players are dead
            if all(player.is_dead for player in population):
                game_paused = True
                
                # GENETIC ALGORITHM: Evolve current generation and intialize new generation
                """
                0. Obtain best solutions from fitness function (here, time_alive)
                1. Select parents (best solutions) as a fraction of the population
                2. Create offspring by updating genes of the rest of the population 
                3. Crossover
                4. Mutation: Apply random mutations according to mutation rate to children
                """
                # for now, we only pass down the best players genes. Later will add option for more parents
                best_solution = population[-1].time_alive
                best_genes = population[-1].genes
                for player in population:
                    player.genes = best_genes

                # apply random mutations too for variability
                if np.random.rand() <= MUTATION_RATE:
                    for player in population:
                        #player.genes += np.random.randint(-10, 10) # TBD: the lower and upper limits should be constants with appropriate name
                        pass
                #reset_game_ga()

                #game_paused = False

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
- (maybe not) Put a lower limit on jump rate 
- Add reset game function: Sets the player and object back to the starting position 
- Implement the genetic algorithm into the game: --> Start by implementing from GA from scratch?
- Add a system that penalizes flying forever OR add a cooldown to jumping
- Add some text in one of the game corners detailing the genetic algorithm status (generation, settings, current best time, ...)
- Add a GUI for starting game and restarting game. Start game will show up when game is first booted, restart game will shwo up when population is dead (not for genetic algorith part though; here it will just reset() the game state with the improved player genes)
"""