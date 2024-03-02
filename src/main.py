import pygame as pg
import numpy as np
import src.common.settings as c
from src.common.player import Player
from src.common.obstacle import Obstacle
from typing import Dict, List, Any
import sys
import copy
import random


# -- Initialize Pygame --
pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((c.WIDTH, c.HEIGHT))
pg.display.set_caption('Obstacle Jumping')


# -- Global Variables --
game_running = True
game_paused = False
game_fps = c.GAME_FPS
generation_clock = 0.0
score = 0
font = pg.font.SysFont(c.FONT_TYPE, c.FONT_SIZE)
fontLarge = pg.font.SysFont(c.FONT_TYPE, c.FONT_SIZE * 2)
info_text = {  # initalize explicitly for clarity with type hints
    'Generation': 1,
    'Best Score': 0,
    'Best Time': 0,
    'Success Rate': Dict[str, Any],
    'k-Success Rate': Dict[str, Any],
    'FPS': game_fps,
}
info_toggle = True
# - Logic -
obstacle_flags: List[bool] = [False] * c.POPULATION_SIZE
# - Data -
best_players = {
    'generation': [],
    'player_id': [],
    'weights_input': [],
    'weights_hidden': [],
    'time_alive': [],
    'highscore': []
}
best_overall_time = 0
best_overall_iw = None
best_overall_hw = None
overall_highscore = 0
overall_deaths = 0
# player scores accessed via player_scores['scores'][player_idx][generation]
player_scores = {
    'player_id': [0] * c.POPULATION_SIZE,
    'scores': [[] for _ in range(c.POPULATION_SIZE)]
}
gen_score = 0
gen_scores = []

# - AI Variables -
population: List[Player] = []
dead_players = []
generation = 1


# -- FUNCTIONS --
def init() -> None:
    if c.is_AI:
        for i in range(c.POPULATION_SIZE):
            population.append(Player(is_AI=True))
            player_scores['player_id'][i] = id(population[i])


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
    # NOTE: Could increase performance a ton by only checking this if players are on ground level
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
    text = fontLarge.render(f"{generation_clock:.1f}", True, c.FONT_COLOR)
    screen.blit(text, text.get_rect(center=(c.WIDTH//2, c.BASE_HEIGHT//2)))


def render_score(screen, score: int) -> None:
    text = fontLarge.render(f"Score: {score}", True, c.FONT_COLOR)
    screen.blit(text, text.get_rect(
        center=(c.WIDTH//2, c.HEIGHT - c.BASE_HEIGHT//2)))


def render_info_text(screen, info: Dict) -> None:
    for n, (k, v) in enumerate(info.items()):
        if k == 'Best Time':
            text = font.render(f"{k}: {v:.2f}", True, c.FONT_INFO_COLOR)
        elif k == 'Success Rate':
            text = font.render(
                f"Average Success Rate: {100*v['mean']:.1f}% ± {100*v['std']:.1f}%", True, c.FONT_INFO_COLOR)
        elif k == 'k-Success Rate':
            if generation >= c.EM_KSUCCESS:
                text = font.render(
                    f"Success Rate (last {c.EM_KSUCCESS} gens): {100*v['mean']:.1f}% ± {100*v['std']:.1f}%", True, c.FONT_INFO_COLOR)
            else:
                text = font.render(
                    f"Success Rate (last {c.EM_KSUCCESS} gens): Awaiting data", True, c.FONT_INFO_COLOR)
        else:
            text = font.render(f"{k}: {v}", True, c.FONT_INFO_COLOR)
        text_x = c.WIDTH - text.get_width() - 20
        text_y = c.BASE_HEIGHT + 20
        y_offset = text_y + n * c.FONT_SIZE
        screen.blit(text, (text_x, y_offset))


def get_success_metric(gen_length: int = generation - 1) -> Dict | None:
    if generation >= gen_length:
        rates = []
        for _ in population:
            rates.append(get_player_success(
                player_id=id(_), gen_length=gen_length))
        result = {
            'mean': np.mean(rates),
            'min': min(rates),
            'max': max(rates),
            'std': np.std(rates)
        }
        return result


def get_player_success(player_id: int, gen_length: int = generation - 1) -> float | None:
    """Evaluation Metric: Player Jump Success Rate
    Computes the fractional obstacle jump success rate for a particular player across generations.

    Args:
        player_id (int): Unique memory address identifier of Player object.
        gen_length (int): Number of past generations to include for evaluating success rate.

    Returns:
        float | None: The obstacle jump success rate of the player.
    """
    if generation >= gen_length:
        if player_id in player_scores['player_id']:
            idx = player_scores['player_id'].index(player_id)
            history = player_scores['scores'][idx][-gen_length:]
            return 1 / (1 + (sum(history) / overall_deaths))
        else:
            print(f"Player ID {player_id} not found")


# - Utility Functions -
def nested_mean(nested_list1, nested_list2):
    """Perform component-wise averaging of two nested lists, provided
    their shapes are identical.

    Args:
        nested_list1: First nested list.
        nested_list2: Second nested list.

    Returns:
        The resulting component-wise averaged nested list.
    """
    return (np.array(nested_list1) + np.array(nested_list2)) / 2


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
                    user_player.jump(fps=game_fps)
                if event.key == pg.K_e:
                    game_fps += 5
                    info_text['FPS'] = game_fps
                if event.key == pg.K_q:
                    game_fps -= 5
                    info_text['FPS'] = game_fps
                if event.key == pg.K_i:
                    info_toggle = not info_toggle

        if not game_paused:
            # - Draw Background Elements + Render Text -
            draw(screen)
            render_timer(screen, generation_clock=generation_clock)
            render_score(screen, score=gen_score)
            if info_toggle:
                render_info_text(screen, info=info_text)

            # - Update Info -
            if gen_score > overall_highscore:
                info_text['Best Score'] = gen_score

            # - Update and Draw Players + Obstacle -
            obstacle.update()
            if obstacle.is_outside():
                obstacle_flags = [False] * c.POPULATION_SIZE

            if c.is_AI:
                display_overlaps(screen, population=population, min_overlaps=2)
                for i, _ in enumerate(population):
                    _.update(obstacle, fps=game_fps)
                    _.draw(screen)
                    if _.x >= + obstacle.x + obstacle.width and not obstacle_flags[i]:
                        _.score += 1
                        obstacle_flags[i] = True
                    if _.score > gen_score:
                        gen_score = _.score
            else:
                user_player.update(obstacle, fps=game_fps)
                user_player.draw(screen)
            obstacle.draw(screen)

            # - Handle Collisions -
            if c.is_AI:
                for i, _ in enumerate(population):
                    if _.is_alive and _.is_colliding(obstacle):
                        _.kill()
                        dead_players.append(_)
                        overall_deaths += 1
                        gen_scores.append(_.score)

                        # normally not needed if order of players in population is not changed:
                        if id(_) in player_scores['player_id']:
                            player_scores['scores'][i].append(_.score)

                if len(dead_players) == c.POPULATION_SIZE:
                    if dead_players[-1].is_animating:  # last dead player
                        pass
                    else:
                        game_running = False
            else:
                if user_player.is_colliding(obstacle):
                    user_player.kill()
                    if user_player.is_animating:
                        pass
                    else:
                        game_paused = True

            pg.display.flip()

        game_tick = clock.tick(game_fps) * (game_fps / 60)
        generation_clock += game_tick / 1000

    else:
        # this is inefficient for large population, but since population would rarely be > 100 it is of limited concern
        # the benefit of this is more clear code

        # NOTE: This is kinda stupid, since dead_players is already sorted by time_alive
        population = sorted(population, key=lambda player: player.fitness())

        best_player = population[-1]
        best_players['generation'].append(generation)
        best_players['weights_input'].append(
            copy.deepcopy(best_player.weights_input))
        best_players['weights_hidden'].append(
            copy.deepcopy(best_player.weights_hidden))
        best_players['time_alive'].append(best_player.time_alive)
        best_players['highscore'].append(max(gen_scores))

        best_overall_index = best_players['time_alive'].index(
            max(best_players['time_alive']))
        best_overall_time = best_players['time_alive'][best_overall_index]
        best_overall_iw = best_players['weights_input'][best_overall_index]
        best_overall_hw = best_players['weights_hidden'][best_overall_index]
        overall_highscore = max(best_players['highscore'])

        # NOTE: With current implementation of reset(), it must be called
        # BEFORE assigning new weights to population
        reset(obstacle=obstacle, players=population)

        # -- Crossover and Mutating --
        for i, _ in enumerate(population):
            # - Standard crossover among KEEP_PARENTS parents -
            if i <= int(len(population) * c.CROSSOVER_RATE):
                parents_iw = random.sample(
                    [population[-j-1].weights_input for j in range(c.KEEP_PARENTS)], 2)
                parents_hw = random.sample(
                    [population[-j-1].weights_hidden for j in range(c.KEEP_PARENTS)], 2)

                child_iw = nested_mean(parents_iw[0], parents_iw[1])
                child_hw = nested_mean(parents_hw[0], parents_hw[1])

                # NOTE: Must deepcopy for nested lists
                _.weights_input = copy.deepcopy(child_iw)
                _.weights_hidden = copy.deepcopy(child_hw)
                _.mutate()

            # - Cross-generation crossover -
            elif i <= int(len(population) * (c.CROSS_GENERATION_RATE + c.CROSSOVER_RATE)):
                parents_iw = [population[-1].weights_input, best_overall_iw]
                parents_hw = [population[-1].weights_hidden, best_overall_hw]

                child_iw = nested_mean(parents_iw[0], parents_iw[1])
                child_hw = nested_mean(parents_hw[0], parents_hw[1])

                _.weights_input = copy.deepcopy(child_iw)
                _.weights_hidden = copy.deepcopy(child_hw)
                _.mutate()

            # - Cloning or Resetting -
            else:
                # randomize if above generation threshold with no performance improvement
                if (generation % c.RESET_THRESHOLD == 0) and (best_players['time_alive'][-c.RESET_THRESHOLD] > best_players['time_alive'][-1]):
                    _.__init__(is_AI=c.is_AI)
                else:
                    _.weights_input = copy.deepcopy(
                        population[-1].weights_input)
                    _.weights_hidden = copy.deepcopy(
                        population[-1].weights_hidden)
                    _.mutate()

        # - Update/Reset Other Elements -
        dead_players = []
        generation_clock = 0.0
        generation += 1
        gen_scores = []
        gen_score = 0
        info_text['Generation'] = generation
        info_text['Best Time'] = best_overall_time
        info_text['Success Rate'] = get_success_metric()
        info_text['k-Success Rate'] = get_success_metric(
            gen_length=c.EM_KSUCCESS)

        game_running = True


"""
List of things to add:
- (1) Storing dictionairy data as .csv before quitting game, along with a copy of constants.py settings. Name files by number starting from 1 and up
- (2!) User-mode and AI-mode toggle (through command line for now; later add UI)
- (3!) Make game even more challenging, e.g. by adding door keys that need to be collected before player can pass through obstacle slit
- (0) Incorporate more complex evaluation metrics like average time survived, time survived distribution for generation + total (plot on screen?)


(!): Challenging
(*): Easy
(n): Priority list
"""
