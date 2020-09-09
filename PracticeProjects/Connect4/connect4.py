
import time
import pygame
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import random as rand
import math

WIDTH = 7
HEIGHT = 6
SPACING = 60
RAD = 20
COLORS = {
    -1: (255, 0, 0),
    0:  (255, 255, 255),
    1:  (0, 0, 255)
}

board_flipped = False
surface = None
board = np.zeros((WIDTH, HEIGHT))

def setupGUI():
    pygame.init()
    surface = pygame.display.set_mode((SPACING*WIDTH, SPACING*HEIGHT))
    pygame.display.set_caption('Connect4')
    return surface

def showGUI(surface=None):
    if surface is None:
        surface = setupGUI()
    if board_flipped:
        thboard = np.flip(board, axis=0)
    else:
        thboard = board
    for x in range(WIDTH):
        for y in range(HEIGHT):
            pygame.draw.circle(
                surface,
                COLORS[thboard.item((x, y))],
                (int((x+.5)*SPACING), int((HEIGHT-y-.5)*SPACING)),
                RAD
            )
    pygame.display.flip()


def build_model():
    model = keras.Sequential([
        keras.layers.ZeroPadding2D(padding=1, input_shape=(WIDTH, HEIGHT, 1)),
        keras.layers.Conv2D(24, (4, 4), activation=None),
        keras.layers.ZeroPadding2D(padding=2),
        keras.layers.Conv2D(64, (4, 4), activation=None),
        keras.layers.ZeroPadding2D(padding=2),
        keras.layers.Conv2D(13, (4, 4), activation=None),
        keras.layers.ZeroPadding2D(padding=1),
        keras.layers.Conv2D(1, (4, 4), activation=None),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # model.build(input_shape=(WIDTH+6, HEIGHT, 2))
    print(model.summary())
    # print(model.layers[1].get_weights())
    return model


def test_win(x, y, my_board=board, side=None):
    if side is None:
        main = my_board.item((x, y))
    else:
        main = side
    dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
    for direc in dirs:
        count = 1
        for sign in (-1, 1):
            tx = x + sign*direc[0]
            ty = y + sign*direc[1]
            stop = False
            while 0 <= tx < WIDTH and 0 <= ty < HEIGHT and not stop:
                if my_board.item((tx, ty)) == main:
                    count += 1
                    tx += sign*direc[0]
                    ty += sign*direc[1]
                else:
                    stop = True
        if count >= 4:
            return True

    return False


def place(value, slot, my_board):
    leval = 0
    winner = 0
    while my_board.item((slot, leval)) != 0:
        leval += 1
        if leval == HEIGHT:
            break
    successfull = leval < HEIGHT
    if successfull:
        my_board.itemset((slot, leval), value)
        if test_win(slot, leval, my_board):
            winner = value
    return successfull, winner


def evaluate_actions(myboard: np.ndarray, model: keras.Model):
    positions = None
    for x in range(WIDTH):
        pos = myboard.copy()
        place(1, x, pos)
        if positions is None:
            positions = pos.reshape((1, WIDTH, HEIGHT, 1))
        else:
            positions = np.append(positions, pos.reshape((1, WIDTH, HEIGHT, 1)), axis=0)

    # print(positions.shape)
    this_predictions = model.predict(positions)
    return this_predictions.reshape(7)


def correct_choice(reg_choice: int, my_board: np.ndarray):
    choice = reg_choice
    importance = 0
    for x in range(7):
        if my_board.item((x, 5)) == 0:
            y = 0
            while my_board.item((x, y)) != 0:
                y += 1
            if test_win(x, y, my_board, 1) and importance < 2:
                choice = x
                importance = 2
            if test_win(x, y, my_board, -1) and importance < 1:
                choice = x
                importance = 1
    return choice


def train_game(history: np.ndarray, model: keras.Model, score=1):
    size = history.shape[0]

    train_for = np.zeros(shape=size)
    for n in range(size):
        train_for.itemset(size-n-1, (score-.5)*math.pow(2/3, n)+.5)

    # train_for.fill(score)
    model.fit(history, train_for, verbose=0)


def play_game_self(red_model: keras.Model, blue_model: keras.Model, delta=1, show=False, training=True, flip=.5, rand_start=False, rand_fuzz=0):
    global board, board_flipped
    board = np.zeros((WIDTH, HEIGHT))
    board_flipped = False
    turn = 0
    if rand_start:
        for v in (-1, 1):
            place(v, rand.randint(0, 6), board)
        turn = 2

    redHistory = np.zeros(shape=(0, WIDTH, HEIGHT, 1))
    blueHistory = np.zeros(shape=(0, WIDTH, HEIGHT, 1))
    History = np.zeros(shape=(0, WIDTH, HEIGHT, 1))
    winner = 0
    while winner == 0 and turn < WIDTH*HEIGHT:
        if flip > rand.random():
            board = np.flip(board, axis=0)
            board_flipped = not board_flipped
        antiBoard = -1*board
        redValues = evaluate_actions(antiBoard, red_model)
        redChoice = np.argmax(redValues)
        if rand_fuzz > rand.random():
            redChoice = rand.randint(0, 6)
        if CHOICE_CORRECT:
            redChoice = correct_choice(redChoice, antiBoard)
        succesfull, winner = place(-1, redChoice, board)
        while not succesfull:
            redValues.itemset(redChoice, -1)
            redChoice = np.argmax(redValues)
            succesfull, winner = place(-1, redChoice, board)

        # redChoices = np.append(arr=redChoices, values=np.array([redChoice]), axis=0)
        redHistory = np.append(arr=redHistory, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
        History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
        if show:
            showGUI(surface)
            time.sleep(delta)

        # print('Red: ')
        # print(redValues)

        turn += 1

        if winner == 0:
            if flip > rand.random():
                board = np.flip(board, axis=0)
                board_flipped = not board_flipped
            blueValues = evaluate_actions(board, blue_model)
            blueChoice = np.argmax(blueValues)
            if rand_fuzz > rand.random():
                blueChoice = rand.randint(0, 6)
            if CHOICE_CORRECT:
                blueChoice = correct_choice(blueChoice, board)
            succesfull, winner = place(1, blueChoice, board)
            while not succesfull:
                blueValues.itemset(blueChoice, -1)
                blueChoice = np.argmax(blueValues)
                succesfull, winner = place(1, blueChoice, board)
            blueHistory = np.append(arr=blueHistory, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
            History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
            if show:
                showGUI(surface)
                time.sleep(delta)
            turn += 1

    if training:
        redHistory = -1*redHistory
        # timePenalty = 1 - History.shape[0]/50
        if winner == 1:  # blue is winner
            train_game(history=redHistory, model=red_model, score=0)
            train_game(history=blueHistory, model=blue_model, score=1)
        if winner == -1:
            train_game(history=redHistory, model=red_model, score=1)
            train_game(history=blueHistory, model=blue_model, score=0)
        if winner == 0:
            train_game(history=redHistory, model=red_model, score=0)
            train_game(history=blueHistory, model=blue_model, score=0)

    time.sleep(.001)

    return winner, turn


def user_choice():
    global surface
    if surface is None:
        showGUI(surface)
    pygame.event.get()
    while not pygame.mouse.get_pressed()[0]:
        pygame.event.get()
        time.sleep(.0001)
    x, y = pygame.mouse.get_pos()
    choice = int(x/SPACING)
    return choice


def play_blue_person(ai_model: keras.Model, delta=1, show=True, training=False):
    global board, board_flipped
    board_flipped = False
    board = np.zeros((WIDTH, HEIGHT))
    History = np.zeros(shape=(0, WIDTH, HEIGHT, 1))
    winner = 0
    turn = 0
    while winner == 0 and turn < WIDTH*HEIGHT:
        antiBoard = -1*board
        redValues = evaluate_actions(antiBoard, ai_model)
        redChoice = np.argmax(redValues)
        if CHOICE_CORRECT:
            redChoice = correct_choice(redChoice, antiBoard)
        succesfull, winner = place(-1, redChoice, board)
        while not succesfull:
            redValues.itemset(redChoice, -1)
            redChoice = np.argmax(redValues)
            succesfull, winner = place(-1, redChoice, board)
        History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
        if show:
            showGUI(surface)
        turn += 1

        if winner == 0:
            blueChoice = user_choice()
            succesfull, winner = place(1, blueChoice, board)
            while not succesfull:
                blueChoice = user_choice()
                succesfull, winner = place(1, blueChoice, board)
            History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)

            if show:
                showGUI(surface)
                time.sleep(delta)

            turn += 1

    if training:
        antiHistory = -1*History
        # timePenalty = 1 - History.shape[0]/50
        if winner == 1:  # blue is winner- player is winner
            train_game(history=antiHistory, model=ai_model, score=0)
            train_game(history=History, model=ai_model, score=1)
        if winner == -1:  # red, AI is winner
            train_game(history=antiHistory, model=ai_model, score=1)
            train_game(history=History, model=ai_model, score=0)
        if winner == 0:
            train_game(history=antiHistory, model=ai_model, score=.5)

    time.sleep(.001)

    return winner, turn


def play_red_person(ai_model: keras.Model, delta=1, show=True, training=False):
    global board, board_flipped
    board_flipped = False
    board = np.zeros((WIDTH, HEIGHT))
    History = np.zeros(shape=(0, WIDTH, HEIGHT, 1))
    winner = 0
    turn = 0
    while winner == 0 and turn < WIDTH*HEIGHT:
        antiBoard = -1*board
        redChoice = user_choice()
        succesfull, winner = place(-1, redChoice, board)
        while not succesfull:
            redChoice = user_choice()
            succesfull, winner = place(-1, redChoice, board)
        History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
        if show:
            showGUI(surface)
            time.sleep(delta)
        turn += 1

        if winner == 0:
            blueValues = evaluate_actions(board, ai_model)
            blueChoice = np.argmax(blueValues)
            if CHOICE_CORRECT:
                blueChoice = correct_choice(blueChoice, board)
            succesfull, winner = place(1, blueChoice, board)
            while not succesfull:
                blueValues.itemset(blueChoice, -1)
                blueChoice = np.argmax(blueValues)
                succesfull, winner = place(1, blueChoice, board)
            History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
            if show:
                showGUI(surface)
                # time.sleep(delta)
            turn += 1

    if training:
        antiHistory = -1*History
        # timePenalty = 1 - History.shape[0]/50
        if winner == 1:  # blue is winner- AI is winner
            train_game(history=antiHistory, model=ai_model, score=0)
            train_game(history=History, model=ai_model, score=1)
        if winner == -1:  # red, user is winner
            train_game(history=antiHistory, model=ai_model, score=1)
            train_game(history=History, model=ai_model, score=0)
        if winner == 0:
            train_game(history=antiHistory, model=ai_model, score=.5)

    time.sleep(.001)

    return winner, turn


def locate(lname: str, version: str):
    return 'saves/'+version+'/'+lname+'_connect4_'+version+'.h5'


# ------------------------------------------------------------------------------------------------------ #

# RED: -1, BLUE: 1

FIRST_TIME = False            # default: FALSE
SAVE = False                  # default: TRUE
USE_OLD_EVAL = True         # default: FALSE
REVERT_TO_CONTROL = False    # default: FALSE

CHOICE_CORRECT = True
RANDOM_START = True
RANDOM_FUZZ = 0.0

TRAIN_COUNT = 000
EVAL_COUNT = 00
TEST_COUNT = 0
PLAY_COUNT = 10

TEST_SPEED = .3
PLAY_SPEED = .4
VERSION = 'NoActive'

players = {

}

control_players = {}



if FIRST_TIME:
    for name in players.keys():
        players[name] = build_model()
        control_players[name] = build_model()
else:
    for name in players.keys():
        if REVERT_TO_CONTROL:
            players[name] = keras.models.load_model(locate('control_'+name, VERSION))
        else:
            players[name] = keras.models.load_model(locate(name, VERSION))
        if USE_OLD_EVAL:
            control_players[name] = keras.models.load_model(locate('control_'+name, VERSION))
        else:
            control_players[name] = keras.models.load_model(locate(name, VERSION))


if TRAIN_COUNT > 0:
    for n in range(TRAIN_COUNT):

        red_player, red_model = rand.choice(list(players.items()))
        blue_player, blue_model = rand.choice(list(players.items()))

        names = {-1: red_player, 0: 'tie\t', 1: blue_player}

        print(str(n) + ': ' + red_player + ' vs. ' + blue_player)

        gwinner, turns = play_game_self(red_model, blue_model, 0, rand_start=RANDOM_START, rand_fuzz=RANDOM_FUZZ)
        print('\tWinner: '+names[gwinner]+'\t\t'+'Turns: '+str(turns))

        if n % 20 == 0:
            if SAVE:
                for player, player_model in players.items():
                    player_model.save(locate(player, VERSION))
                print('saving...')


if SAVE:
    for player, player_model in players.items():
        player_model.save(locate(player, VERSION))
    print('all models saved to memory')


if EVAL_COUNT > 0:
    control_wins = 0
    trained_wins = 0
    ties = 0
    for n in range(EVAL_COUNT):

        trained_player, trained_model = rand.choice(list(players.items()))
        control_player, control_model = rand.choice(list(control_players.items()))
        trained_player = 't:'+trained_player
        control_player = 'c:'+control_player
        if rand.random() > .5:
            names = {-1: trained_player, 0: 'tie\t', 1: control_player}
            red_model = trained_model
            blue_model = control_model
        else:
            names = {1: trained_player, 0: 'tie\t', -1: control_player}
            blue_model = trained_model
            red_model = control_model
        print('eval - '+str(n) + ': ' + names[-1] + ' vs. ' + names[1])
        gwinner, turns = play_game_self(red_model, blue_model, 0, training=False, show=False)
        print('\tWinner: '+names[gwinner]+'\t\t'+'Turns: '+str(turns))
        if names[gwinner] == trained_player:
            trained_wins += 1
        if names[gwinner] == control_player:
            control_wins += 1
        if gwinner == 0:
            ties += 1

    growth_factor = trained_wins / control_wins
    CRITICAL_Z = 1.96  # 95% confidence
    marginal_error = CRITICAL_Z * math.sqrt(.25/(trained_wins+control_wins))
    print('\nResults----------------------------------------')
    print('TOTAL EVAL GAMES: '+str(EVAL_COUNT))
    print('\t| trained_wins\t| control_wins\t| ties')
    print('\t|\t'+str(trained_wins)+'\t\t\t|\t '+str(control_wins)+'\t\t\t|\t '+str(ties))
    print('IMPROVEMENT_FACTOR: '+str(round(growth_factor, 3)))
    print('PERFORMANCE_CHANGE: ' + str(round((growth_factor-1) * 100, 1)) + '%')
    print('SAMPLE_INDUCED_MAX_MARGINAL_ERROR: +/- '+str(round(100*marginal_error, 1))+'%\n\t*at 95% confidence')
    print('-------------------------------------------------\n')

if TEST_COUNT > 0:
    for n in range(TEST_COUNT):

        red_player, red_model = rand.choice(list(players.items()))
        blue_player, blue_model = rand.choice(list(players.items()))

        names = {-1: red_player, 0: 'tie', 1: blue_player}

        print('test - '+str(n) + ': ' + red_player + ' vs ' + blue_player)

        gwinner, turns = play_game_self(red_model, blue_model, TEST_SPEED, show=True, training=True)
        print('\tWinner: '+names[gwinner]+'\t\t'+'Turns: '+str(turns))


if SAVE:
    for player, player_model in players.items():
        player_model.save(locate(player, VERSION))
    for c_player, c_model in control_players.items():
        c_model.save(locate('control_'+c_player, VERSION))
    print('all models saved to memory')


if PLAY_COUNT > 0:
    for n in range(PLAY_COUNT):

        red_player, red_model = rand.choice(list(players.items()))

        if rand.random() > .5:
            names = {-1: red_player, 0: 'tie', 1: 'User'}
            print('test - ' + str(n) + ': ' + red_player + ' vs User')
            gwinner, turns = play_blue_person(red_model, 1, show=True, training=False)
        else:
            names = {1: red_player, 0: 'tie', -1: 'User'}
            print('test - ' + str(n) + ': User vs. ' + red_player)
            gwinner, turns = play_red_person(red_model, 1, show=True, training=False)
        print('\tWinner: '+names[gwinner]+'\t\t'+'Turns: '+str(turns))






