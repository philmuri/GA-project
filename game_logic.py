import pygame as pg
import numpy as np
import time
import sys
from typing import List, Dict

# Constants: Game
WIDTH, HEIGHT = 800, 600
BASE_HEIGHT = 50

OBSTACLE_WIDTH_MIN = 50
OBSTACLE_WIDTH_MAX = 100
OBSTACLE_HEIGHT_MIN = 50
OBSTACLE_HEIGHT_MAX = 300

PLAYER_START_POSITION = 80
PLAYER_RADIUS = 20
PLAYER_JUMP_COOLDOWN = 0.25

BG_COLOR = (200, 200, 200)
BASE_COLOR = (0, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)
PLAYER_COLOR = (128, 128, 128)
PLAYER_DEATH_COLOR = (255, 0, 0)
FONT_COLOR = (128, 128, 128)
FONT_SIZE = 16
FONT_TYPE = 'Times New Roman'

OBSTACLE_SPEED = 10
GRAVITY = 0.5
JUMP_FORCE = -10

GAME_FPS = 120

# Constants: Genetic Algorithm
POPULATION_SIZE = 30
MUTATION_RATE = 0.1
MUTATION_SIZE_FACTOR = 1


class Player:
    def __init__(self, genes=None):
        self.radius = PLAYER_RADIUS
        self.x = PLAYER_START_POSITION
        self.y = HEIGHT - BASE_HEIGHT - self.radius
        self.vy = 0
        self.is_jumping = False
        self.jump_cooldown = PLAYER_JUMP_COOLDOWN
        self.last_jump_time = 0
        self.color = PLAYER_COLOR
        self.genes = genes # for now its content are not explicit, but it must always be structured as a List[int] where [dist_rule, height_rule, jumpforce_rule] 
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
        Jump event for player. Jump force is a gene evolved with genetic algorithm if player has genes.
        """
        self.is_jumping = True
        if self.genes:
            self.vy = self.genes[2]
        else:
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


def save_data(filename: str, data):
    filename = filename + '.txt'
    with open(filename,'w') as file:
        for l in data:
            file.write(str(l) + '\n')


def render_info_text(screen, states: Dict, x: int, y: int):
    """
    states: A dictionairy object containing state names as keys and their content as values
    """
    font = pg.font.SysFont(FONT_TYPE, FONT_SIZE)
    for n, (k, v) in enumerate(states.items()):
        text = font.render(f"{k}: {v}", True, FONT_COLOR) # Text
        y_offset = y + n * FONT_SIZE
        screen.blit(text, (x, y_offset))



# -- GENETIC ALGORITHM GAME --
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
        jump_power = int(np.random.uniform(-20, -5))
        init_genes = [dist_threshold, height_threshold, jump_power]
        player = Player(genes=init_genes)
        population.append(player)
    # - Initialize obstacle
    obstacle = Obstacle()
    # - Data collecton
    n_generation = 0
    best_solution = 0
    previous_best_solution = 0
    overall_best_solution = 0
    best_solution_times = []
    genes_over_generations = []
    times_over_generations = []
    ga_states = {"Generation": 0, 
                 "Best Time": 0, 
                 "Previous Best Time": 0,
                 "Overall Best Time": 0}

    # -- Run game --
    game_running = True
    game_paused = False
    while game_running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False

                save_data('best_solution_times', best_solution_times)
                save_data('genes_over_generations', genes_over_generations)
                save_data('times_over_generations', times_over_generations)

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                        game_paused = not game_paused


        if not game_paused:
            # - Draw game elements
            screen.fill(BG_COLOR)
            pg.draw.rect(screen, BASE_COLOR, (0, HEIGHT - BASE_HEIGHT, WIDTH, HEIGHT)) # Ground
            pg.draw.rect(screen, BASE_COLOR, (0, 0, WIDTH, BASE_HEIGHT)) # Roof
            render_info_text(screen, ga_states, WIDTH - 180, BASE_HEIGHT + 5) # magic numbers yes
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
                
                # - GENETIC ALGORITHM: Evolve current generation and intialize new generation -
                # for now, we only pass down the best players genes. Later will add option for more parents
                best_solution = population[-1].time_alive
                if n_generation > 0: previous_best_solution = best_solution_times[-1][0]
                best_genes = population[-1].genes
                
                # store data
                best_solution_times.append([best_solution, best_genes])
                genes_over_generations.append([player.genes[:] for player in population])
                times_over_generations.append([player.time_alive for player in population])
                overall_best_solution = max([time[0] for time in best_solution_times])
                print(overall_best_solution)
                
                for player in population:
                    player.genes = best_genes

                # - Random mutations
                mutated_genes = []
                for player in population:
                    for n in range(len(player.genes)):
                        if np.random.rand() <= MUTATION_RATE:
                            if n <= 1:
                                player.genes[n] += int(np.random.normal(0, 2 * MUTATION_SIZE_FACTOR)) # TBD: the lower and upper limits should be constants with appropriate name
                            if n == 2:
                                player.genes[n] += int(np.random.normal(0, 1 * MUTATION_SIZE_FACTOR)) # 1 st.dev. is roughly [-4, 4]
                    mutated_genes.append(player.genes[:])

                print(f"Generation: {n_generation} | Best solution time: {best_solution} (previous: {previous_best_solution}) | Genes: {best_genes}")
                ga_states["Generation"] = n_generation
                ga_states["Best Time"] = best_solution
                ga_states["Previous Best Time"] = previous_best_solution
                ga_states["Overall Best Time"] = overall_best_solution

                obstacle.__init__()
                for n, player in enumerate(population):
                    player.__init__(mutated_genes[n])

                n_generation += 1

                game_paused = False

            pg.display.flip()

        clock.tick(GAME_FPS)

    pg.quit()
    sys.exit()




# -- STANDARD GAME --
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
                time.sleep(1)
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
- DATA: Create new folder for data each game run with data if number of generations is significantly large, and also store info like top X best overall times and the corresponding genes
- Add a system that penalizes flying forever + make jump cooldown another gene?
- ! Add some text in one of the game corners detailing the genetic algorithm status (generation, settings, current best time, round timer ...)
- Add a GUI for starting game and restarting game. Start game will show up when game is first booted, restart game will shwo up when population is dead (not for genetic algorith part though; here it will just reset() the game state with the improved player genes)
- ! ADD another feature to combat degradation effect of mutations on evolution: 
    Generations survived as another performance variable. Make a new attribute which counts and adds one to itself each generation.
    The nature of the new fitness function could e.g. be the product generations_survived * time_alive.
    NOTE: This will require implementing keeping multiple parents; otherwise the same player would be favored forever
- Consider what other data can be saved for analysis
"""