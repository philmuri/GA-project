import pygame as pg
# import numpy as np
# import time
# import sys
import constants as c
from player import Player
from obstacle import Obstacle


init_input_genes = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0]]  # for now 5 neurons
init_hidden_genes = [[0], [0], [0]]  # 3 neuron hidden layer

# Global Variables
game_running = True
game_paused = False
population = []
dead_players = 0

round_time = 0
score = 0
# font = pg.font.SysFont(FONT_TYPE, FONT_SIZE)
# fontLarge = pg.font.SysFont(FONT_TYPE, FONT_SIZE * 2)

generation = 1

# -- Initialize Pygame --
pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((c.WIDTH, c.HEIGHT))
pg.display.set_caption('Obstacle Jumping')


# -- FUNCTIONS --
def init() -> None:
    if c.is_AI:
        for _ in range(c.POPULATION_SIZE):
            population.append(Player(is_AI=True))


def restart():
    # re-initailize obstacles and players
    # genetic algorithm: upddate players with new weights andd mutations
    # keep 2 best players without mutating them
    # OR keep best player of current generation and best player overall
    # 1/3 of population is cloned from top player in generation and mutated
    # 1/3 of population is bred from top player and overall top player and mutated
    # 1/3 - 2 of population is respawned if fitness score hasnt increased in some number of generations, else bred and mutated from top player and second best player

    pass


def draw(screen) -> None:
    screen.fill(c.BG_COLOR)
    pg.draw.rect(screen, c.BASE_COLOR, (0, c.HEIGHT -
                 c.BASE_HEIGHT, c.WIDTH, c.HEIGHT))  # Ground
    pg.draw.rect(screen, c.BASE_COLOR, (0, 0, c.WIDTH, c.BASE_HEIGHT))  # Roof
    pass


def render_timer(screen, round_time: float) -> None:
    font = pg.font.SysFont(c.FONT_TYPE, c.FONT_SIZE * 2)
    text = font.render(f"{round_time:.1f}", True, c.FONT_COLOR)
    screen.blit(text, text.get_rect(center=(c.WIDTH//2, c.BASE_HEIGHT//2)))


# -- Main Game Loop --
init()
user_player = Player()
obstacle = Obstacle()

while game_running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            game_running = False
            # NOTE: Save data at this stage
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_p:
                game_paused = not game_paused
            if event.key == pg.K_SPACE:
                user_player.jump()

    if not game_paused:
        # - Draw Background Elements + Render Text -
        draw(screen)
        render_timer(screen, round_time=round_time)

        # - Update and Draw Players + Obstacle -
        obstacle.update()
        if c.is_AI:
            for _ in population:
                _.update()
                _.draw(screen)
        else:
            user_player.update()
            user_player.draw(screen)
        obstacle.draw(screen)

        # - Handle Collisions -
        if c.is_AI:
            for _ in population:
                if _.is_alive and _.is_colliding(obstacle=obstacle):
                    _.is_alive = False
                    dead_players += 1
        else:
            if user_player.is_colliding(obstacle=obstacle):
                game_paused = True

        # - Handle all players dead -
        # for player in population:
        #    if player.is_alive:
        # handle player-object collision
        #        pass

        # if dead_players == len(POPULATION_SIZE):
        #    game_running = False

        pg.display.flip()

    game_tick = clock.tick(c.GAME_FPS)
    round_time += game_tick / 1000

else:
    best_players = []
    print(c.BASE_COLOR, c.OBSTACLE_COLOR, c.BG_COLOR)
    # 1) obtain best two players (id or from index in population) and best fitness score

    # 2) keep the best bird, and extract their inputweights and hiddenweights using deepcopy from copy library
    # bestInputWeights = copy.deepcopy(population[best_player].inputWeights)
    # bestHiddenWeights = copy.deepcopy(population[best_player].hiddenWeights)

    # 3) Update new highscores (best generation, best score, best fitness)

    # 4) Store the players to breed in their separate list using deecopy again
    # best_players.append(copy.deepcopy(population[best_player]))
    # population.pop(i)
