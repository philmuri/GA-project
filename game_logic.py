import pygame as pg
import numpy as np
import time
import sys
import heapq
from typing import List, Dict, Any
from pathlib import Path

# Global Variables
WIDTH = 1200
HEIGHT = 600
BASE_HEIGHT = 50

OBSTACLE_WIDTH_MIN = 100
OBSTACLE_WIDTH_MAX = 150
OBSTACLE_HEIGHT_MIN = 50
OBSTACLE_HEIGHT_MAX = 300

PLAYER_START_POSITION = 200
PLAYER_RADIUS = 25
PLAYER_JUMP_COOLDOWN = 0.25

BG_COLOR = (200, 200, 200)
BASE_COLOR = (0, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)
PLAYER_COLOR = (128, 128, 128)
PLAYER_DEATH_COLOR = (255, 0, 0)
FONT_COLOR = (128, 128, 128)
FONT_SIZE = 16
FONT_TYPE = 'Calibri'

OBSTACLE_SPEED = 10
GRAVITY = 0.5
JUMP_FORCE = -10

GAME_FPS = 90

# Global Variables: GA
POPULATION_SIZE = 30
MUTATION_RATE = 0.5
MUTATION_SIZE_FACTOR = 1
KEEP_PARENTS = 2


class Player:
    def __init__(self, genes=None, toughness=None):
        self.radius = PLAYER_RADIUS
        self.x = PLAYER_START_POSITION
        self.y = HEIGHT - BASE_HEIGHT - self.radius
        self.vy = 0
        self.is_jumping = False
        self.jump_cooldown = PLAYER_JUMP_COOLDOWN
        self.last_jump_time = 0
        self.color = PLAYER_COLOR
        # for now its content are not explicit, but it must always be structured as a List[int] where [dist_rule, height_rule, jumpforce_rule]
        self.genes = genes
        self.init_time = time.time()
        self.is_animating = False
        self.animation_start_time = 0
        self.is_dead = False
        # Performance attributes:
        self.time_alive = 0
        self.toughness = toughness

    def draw(self, screen, color):
        """
        Draw the player in its current position or according to an animation.
        """
        if self.is_animating:
            self.animation(screen)  # separate draw animation
        else:
            pg.draw.circle(screen, color, (self.x, self.y), self.radius)

    def update(self, obstacle=None):
        """
        Handle player motion by updating the player y-position according to simple equation of motion.
        Also handles genetic algorithm player motion by calling jump() when should_jump() is True.
        """
        if self.is_animating:
            pass  # disable physics on animating player
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
        if self.y >= HEIGHT - BASE_HEIGHT - self.radius and self.vy >= 0:  # if player near (or under) ground while having downward or no velocity, terminate jump and reset player to ground level with zero velocity
            self.y = HEIGHT - BASE_HEIGHT - self.radius
            self.vy = 0
            self.is_jumping = False
        # if in air and upward velocity, act with gravity (downward constant acceleration)
        else:
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
        dx = self.x - max(obstacle.x, min(self.x, obstacle.x + obstacle.width)
                          )  # a smart way to get the nearest x-coordinate of the obstacle to the player's center
        dy = self.y - max(obstacle.y, min(self.y,
                          obstacle.y + obstacle.height))
        dist = dx**2 + dy**2
        return dist < self.radius**2 or self.y <= BASE_HEIGHT  # also handles roof collision

    def should_jump(self, obstacle):  # for GA training only
        """
        Jump logic for genetic algorithm training, based on player proximity to obstacle in x and y direction (separately).
        This introduces functionality to the genes in the player chromosome and is the basis for the evolution process.
        (NOTE: for genetic algorithm part only)
        """
        dist = obstacle.x - \
            self.x  # + when left of obstacle, - right when player center crossing obstacle
        # + when below obstacle, - when player center crossing above obstacle
        height_diff = self.y - obstacle.y
        # player ignores current obstacle distance once the center passes its right side. player jumps before then only if also its distance to obstacle is less than the gene-specified value
        if (dist + obstacle.width >= 0 and dist <= self.genes[0]):
            return True
        # if player is sufficiently below an obstacle, jump
        if (height_diff >= self.genes[1]):
            return True
        return False

    def kill(self):  # if kill is called, player object enters is_animating state and will cease to update() while draw() has a custom behavior (in this case moving with obstacle speed)
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
            pg.draw.circle(screen, PLAYER_DEATH_COLOR,
                           (self.x, self.y), self.radius)

    def fitness_score(self):
        """
        Fitness function for the genetic algorithm.
        Currently modulated by the toughness (generations survived) of the player, with initially a large contribution which decays the longer the player has survived.
        The idea is to favor the surviving player as a parent/solution for a little while longer than one generation on average, so as to counteract the randomness from
        mutations. As the generations pass and the offspring starts to outperform the surviving player in time_alive, their fitness score should drop to give them less importance.
        """
        if self.toughness > 0:
            return self.time_alive * (1 / self.toughness + 1)
        else:
            return self.time_alive


class Obstacle:
    def __init__(self):  # obstacle height will be randomized
        self.width = np.random.randint(OBSTACLE_WIDTH_MIN, OBSTACLE_WIDTH_MAX)
        self.height = np.random.randint(
            OBSTACLE_HEIGHT_MIN, OBSTACLE_HEIGHT_MAX)
        self.x = WIDTH
        self.y = HEIGHT - BASE_HEIGHT - self.height

    def draw(self, screen):
        pg.draw.rect(screen, OBSTACLE_COLOR,
                     (self.x, self.y, self.width, self.height))

    def update(self):
        if self.x + self.width <= 0:
            self.__init__()  # this might be bad practice
        self.x -= OBSTACLE_SPEED


# list lengths must match
def save_data(file_names: List[str], data: List[Any], folder_name: str = 'GA_data'):
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)

    for n, file_name in enumerate(file_names):
        with open(folder_path / file_name, 'w') as file:
            for line in data[n]:
                file.write(str(line) + '\n')


def render_info_text(screen, states: Dict, x: int, y: int):
    """
    states: A dictionairy object containing state names as keys and their content as values
    """
    font = pg.font.SysFont(FONT_TYPE, FONT_SIZE)
    for n, (k, v) in enumerate(states.items()):
        text = font.render(f"{k}: {v:.1f}", True, FONT_COLOR)  # Text
        y_offset = y + n * FONT_SIZE
        screen.blit(text, (x, y_offset))


def render_timer(screen, round_time: float, x: int, y: int):
    font = pg.font.SysFont(FONT_TYPE, FONT_SIZE * 2)
    text = font.render(f"{round_time:.1f}", True, FONT_COLOR)
    screen.blit(text, text.get_rect(center=(WIDTH//2, BASE_HEIGHT//2)))


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
        player = Player(genes=init_genes, toughness=0)
        population.append(player)
    # - Initialize obstacle
    obstacle = Obstacle()
    # - Data collecton
    round_time = 0
    n_generation = 0
    best_times = 0
    previous_best_solution = 0
    overall_best_solution = 0
    best_solution_times = []
    genes_over_generations = []
    times_over_generations = []
    toughness_over_generations = []
    ga_states = {"Generation": 0,
                 "Best Time": 0,
                 "Previous Best Time": 0,
                 "Overall Best Time": 0,
                 "Highest Toughness": 0}

    # -- Run game --
    game_running = True
    game_paused = False
    while game_running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False

                save_data(file_names=['best_solution_times.txt', 'genes_over_generations.txt', 'times_over_generations.txt', 'toughness_over_generations.txt'],
                          data=[best_solution_times, genes_over_generations, times_over_generations, toughness_over_generations])

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    game_paused = not game_paused

        if not game_paused:
            # - Draw game elements each game tick
            screen.fill(BG_COLOR)
            pg.draw.rect(screen, BASE_COLOR, (0, HEIGHT -
                         BASE_HEIGHT, WIDTH, HEIGHT))  # Ground
            pg.draw.rect(screen, BASE_COLOR,
                         (0, 0, WIDTH, BASE_HEIGHT))  # Roof
            render_info_text(screen, ga_states, WIDTH - 180,
                             BASE_HEIGHT + 5)  # magic numbers yes
            render_timer(screen, round_time, 0, 0)
            # - Update players and obstacle
            for player in population:
                player.update(obstacle)
            obstacle.update()
            # - Handle player-obstacle collision events
            for player in population:
                if not player.is_dead and player.is_colliding(obstacle=obstacle):
                    player.kill()

            # - Redraw players and obstacle each game tick
            for player in population:
                player.draw(screen, player.color)
            obstacle.draw(screen)

            # - Handle case when all players are dead
            if all(player.is_dead for player in population):

                # -- GENETIC ALGORITHM: Evolve current generation and intialize new generation --
                best_players = heapq.nlargest(
                    KEEP_PARENTS, population, key=lambda player: (player.fitness_score()))
                best_times = [
                    best_player.time_alive for best_player in best_players]
                best_genes = [
                    best_player.genes for best_player in best_players]

                # - Update all player toughness (NOTE: Temporary implementation until I find a 'cleaner' way)
                new_toughness = [0] * len(population)
                for i, player in enumerate(population):
                    if player in best_players:
                        player.toughness += 1
                    else:
                        player.toughness = 0
                    new_toughness[i] = player.toughness

                # - Store data
                best_solution_times.append(
                    [n_generation, id(best_players), best_times, best_genes])
                genes_over_generations.append(
                    [n_generation, [[id(player), player.genes[:]] for player in population]])
                times_over_generations.append(
                    [n_generation, [[id(player), player.time_alive] for player in population]])
                toughness_over_generations.append(
                    [n_generation, [[id(player), player.toughness] for player in population]])

                # - Update text display info variables
                if n_generation > 0:
                    previous_best_solution = max(best_solution_times[-2][2])
                overall_best_solution = max(
                    [max(solution[2]) for solution in best_solution_times])
                ga_states["Generation"] = n_generation
                ga_states["Best Time"] = max(best_times)
                ga_states["Previous Best Time"] = previous_best_solution
                ga_states["Overall Best Time"] = overall_best_solution
                ga_states["Highest Toughness"] = max(new_toughness)

                # - Crossover
                for player in population:
                    # use average for chromosome crossover
                    player.genes = list(np.mean(np.array(best_genes), axis=0))

                # - Mutations
                mutated_genes = []
                for player in population:
                    for n in range(len(player.genes)):
                        if np.random.rand() <= MUTATION_RATE:
                            if n <= 1:
                                # TBD: the lower and upper limits should be constants with appropriate name
                                player.genes[n] += int(np.random.normal(0,
                                                       2 * MUTATION_SIZE_FACTOR))
                            if n == 2:
                                # 1 st.dev. is roughly [-4, 4]
                                player.genes[n] += int(np.random.normal(0,
                                                       1 * MUTATION_SIZE_FACTOR))
                    mutated_genes.append(player.genes[:])

                # - Reset attributes (NOTE: dirty method for now; TBD: add reset methods to each class that properly and safely handle attribute resetting)
                obstacle.__init__()
                for n, player in enumerate(population):
                    player.__init__(mutated_genes[n], new_toughness[n])

                n_generation += 1
                round_time = 0

            pg.display.flip()

        game_tick = clock.tick(GAME_FPS)
        round_time += game_tick / 1000

    pg.quit()
    sys.exit()


if __name__ == '__main__':
    run_game_ga()


"""
TBD:
- Refactoring:
    - Make a class for genetic algorithm part?
    - Lots of repeated loops that can potentially be grouped together
    - For cleaner code, changing 'genes' attribute to be a dict might be better
- Error handling:
    - KEEP_PARENTS <= POPULATION_SIZE
    - Negative inputs and general non-physical inputs can be treated later (if ever...)


- IMPORTANT: Add another gene being the height to the ceiling
- IMPORTANT: If this still doesn't train properly, replace current fitness function with actual neural network structure. See https://github.com/felschatz/Sloppy-Block/blob/master/bird.py

- Initialize players with a unique random color tied to their memory address (accessible with id(player))
- DATA: Define function that creates folder and stores a csv style file (using pandas) containing the following columns for EACH player ID (us eid() for unique identifier from memory address):
    - Generations: Current generation (row 1 is generation 1, row 2 is 2 and so on)
    - Genes: A list of all gene values in the usual order [dist, height, jumpforce]
    - Toughess: Toughness of the player in current generation
    - Fitness score: Depending on how its defined, can be derived from Genes and Toughness, so it might be redundant
    - NOTE: Alongside the data should be all the base parameters for genetic algorithm plus game settings, e.g. OBSTACLE_SPEED, KEEP_PARENTS and so on. Perhaps in a seperate file titled 'settings', possibly in yaml
- DATA: Other things that can be stored in a single data file (although this can be derived from the latter-mentioned data)
    - Best solution time overall
    - Highest toughness overall
    - Highest value of fitness function
- Add a system that penalizes flying forever + make jump cooldown another gene?  --> YES
- Add a GUI for starting game and restarting game. Start game will show up when game is first booted, restart game will shwo up when population is dead (not for genetic algorith part though; here it will just reset() the game state with the improved player genes)
- Make obstacles more difficult. Perhaps adopt the flappy bird style of top and bottom columns with a fixed size for the hole. This would require updating the Obstacle draw method

"""
