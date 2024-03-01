import pygame as pg
import numpy as np
from heapq import nlargest
import constants as c
from player import Player
from obstacle import Obstacle
from typing import Dict, List
import sys
import copy
import time


# -- Initialize Pygame --
pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((c.WIDTH, c.HEIGHT))
pg.display.set_caption('Obstacle Jumping')


# -- Global Variables --
game_running = True
game_paused = False
generation_clock = 0.0
score = 0
font = pg.font.SysFont(c.FONT_TYPE, c.FONT_SIZE)
fontLarge = pg.font.SysFont(c.FONT_TYPE, c.FONT_SIZE * 2)
info_text = {
    'Generation': 0,
    'Best Time': 0,
}
# - Data -
all_best_times = []
best_overall_time = 0
# - AI Variables -
population: List[Player] = []
dead_players = 0
generation = 1
init_input_genes = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0]]  # for now 5 neurons
init_hidden_genes = [[0], [0], [0]]  # 3 neuron hidden layer


# -- FUNCTIONS --
def init() -> None:
    if c.is_AI:
        for _ in range(c.POPULATION_SIZE):
            population.append(Player(is_AI=True))


def reset(obstacle: Obstacle, players: List[Player] | Player) -> None:
    obstacle.__init__()
    if c.is_AI and isinstance(players, List):
        for _ in players:
            _.__init__(is_AI=c.is_AI)
    else:
        players.__init__()

    # NOTE: Later replace this implementation with removing Player instances and creating new ones?


def draw(screen) -> None:
    screen.fill(c.BG_COLOR)
    pg.draw.rect(screen, c.BASE_COLOR, (0, c.HEIGHT -
                 c.BASE_HEIGHT, c.WIDTH, c.HEIGHT))  # Ground
    pg.draw.rect(screen, c.BASE_COLOR, (0, 0, c.WIDTH, c.BASE_HEIGHT))  # Roof
    pass


def check_overlap(player1: Player, player2: Player) -> bool:
    return all([player1.y == player2.y])


def display_overlaps(screen, population: List[Player], min_overlaps: int) -> None:
    # NOTE: This is ~O(n^2) (no overlaps) and up to ~O(n^3) (many overlaps)
    overlap_groups = []
    text_pos_registry = set()
    for i in range(len(population)):
        overlap_group = [population[i]]

        for j in range(i+1, len(population)):
            if check_overlap(population[i], population[j]):
                overlap_group.append(population[j])

        if len(overlap_group) > min_overlaps:
            overlap_groups.append(overlap_group)

    for group in overlap_groups:
        text_pos = (group[0].x - 2.5 * group[0].radius,
                    group[0].y)
        if text_pos not in text_pos_registry:
            text = font.render(f"x{len(group)}", True, c.OBSTACLE_COLOR)
            screen.blit(text, text_pos)
            text_pos_registry.add(text_pos)


def render_timer(screen, generation_clock: float) -> None:
    font = pg.font.SysFont(c.FONT_TYPE, c.FONT_SIZE * 2)
    text = font.render(f"{generation_clock:.1f}", True, c.FONT_COLOR)
    screen.blit(text, text.get_rect(center=(c.WIDTH//2, c.BASE_HEIGHT//2)))


def render_info_text(screen, info: Dict) -> None:
    for n, (k, v) in enumerate(info.items()):
        if isinstance(v, float):
            text = font.render(f"{k}: {v:.2f}", True, c.FONT_COLOR)
        else:
            text = font.render(f"{k}: {v}", True, c.FONT_COLOR)
        text_x = c.WIDTH - text.get_width() - 20
        text_y = c.BASE_HEIGHT + 20
        y_offset = text_y + n * c.FONT_SIZE
        screen.blit(text, (text_x, y_offset))


# -- Main Game Loop --
init()
user_player = Player()
obstacle = Obstacle()

while True:
    if game_running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False
                # NOTE: Save data at this stage
                pg.quit()
                sys.exit()

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    game_paused = not game_paused
                if event.key == pg.K_SPACE:
                    user_player.jump()

        if not game_paused:
            # - Draw Background Elements + Render Text -
            draw(screen)
            render_timer(screen, generation_clock=generation_clock)
            render_info_text(screen, info=info_text)

            # - Update and Draw Players + Obstacle -
            obstacle.update()
            if c.is_AI:
                display_overlaps(screen, population=population, min_overlaps=2)
                for _ in population:
                    _.update(obstacle)
                    _.draw(screen)
            else:
                user_player.update(obstacle)
                user_player.draw(screen)
            obstacle.draw(screen)
            # - Handle Collisions -
            if c.is_AI:
                for _ in population:
                    if _.is_alive and _.is_colliding(obstacle):
                        _.kill()
                        dead_players += 1
                if dead_players == c.POPULATION_SIZE:
                    game_running = False
            else:
                if user_player.is_colliding(obstacle):
                    game_paused = True
                    user_player.kill()

            pg.display.flip()

        game_tick = clock.tick(c.GAME_FPS)
        generation_clock += game_tick / 1000

    else:
        # this is inefficient for large population, but since population would rarely be > 100 it is of limited concern
        # the benefit of this is more clear code
        population = sorted(population, key=lambda player: player.fitness())

        best_time = population[-1].time_alive
        all_best_times.append(best_time)
        best_overall_time = max(all_best_times)

        best_input_weights = population[-1].weights_input
        best_hidden_weights = population[-1].weights_hidden

        print(f"Best input weights:\n {best_input_weights}")
        print(f"Best hidden weights:\n {best_hidden_weights}")

        # NOTE: With current implementation of reset(), it must be called
        # BEFORE assigning new weights to population
        reset(obstacle=obstacle, players=population)

        # NOTE: For now, simple cloning + mutation of best player in generation
        for _ in population:
            _.weights_input = copy.deepcopy(best_input_weights)
            _.weights_hidden = copy.deepcopy(best_hidden_weights)
            _.mutate()

        time.sleep(0.5)
        dead_players = 0
        generation_clock = 0.0
        game_running = True

# 2) keep the best bird, and extract their inputweights and hiddenweights using deepcopy from copy library
# bestInputWeights = copy.deepcopy(population[best_player].inputWeights)
# bestHiddenWeights = copy.deepcopy(population[best_player].hiddenWeights)

# 3) Update new highscores (best generation, best score, best fitness)

# 4) Store the players to breed in their separate list using deecopy again
# best_players.append(copy.deepcopy(population[best_player]))
# population.pop(i)
