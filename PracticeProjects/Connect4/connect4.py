
import time
import pygame
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import random as rand
import math
import os

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


def more_dense_sigs_model():
    model = keras.Sequential([
        keras.layers.ZeroPadding2D(padding=1, input_shape=(WIDTH, HEIGHT, 1)),
        keras.layers.Conv2D(50, (4, 4), activation='sigmoid'),
        keras.layers.ZeroPadding2D(padding=2),
        keras.layers.Conv2D(200, (4, 4), activation='sigmoid'),
        keras.layers.Flatten(),
        keras.layers.Dense(200, activation='sigmoid'),
        keras.layers.Dense(60, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
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
    successes = []
    for x in range(WIDTH):
        pos = myboard.copy()
        sucessfull, winner = place(1, x, pos)
        successes.append(sucessfull)
        if positions is None:
            positions = pos.reshape((1, WIDTH, HEIGHT, 1))
        else:
            positions = np.append(positions, pos.reshape((1, WIDTH, HEIGHT, 1)), axis=0)

    this_predictions = model.predict(positions)  # predictions are between 0 and 1

    return successes,  this_predictions.reshape(7)


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


class Player:
    def __init__(self, name, trainable):
        self.trainable = trainable
        self.name = name

    def get_choice(self, game_board):
        pass

    def train(self, game_history, score):  # x is the input group, score
        pass


class ModelPlayer(Player):
    def __init__(self, name, model, is_training, choice_correct):
        super().__init__(name, is_training)
        self.model = model
        self.choice_correct = choice_correct

    def get_choice(self, eval_board):
        sucesses, values = evaluate_actions(eval_board, self.model)  # get a list of confidences in choice
        choice = np.argmax(values)  # pick the one with the strongest confidence
        while not sucesses[choice]:
            values.itemset(choice, 0)
            choice = np.argmax(values)
        if self.choice_correct:  # if choice correct is on, automatically fix errors
            choice = correct_choice(choice, eval_board)
        return choice

    def train(self, game_history, score):  # x is the input group, score
        if self.trainable:
            train_game(history=game_history, model=self.model, score=score)


class UserPlayer(Player):
    def __init__(self, name):
        super().__init__(name=name, trainable=False)

    def get_choice(self, eval_board):
        global surface
        if surface is None:
            showGUI(surface)
        pygame.event.get()
        while not pygame.mouse.get_pressed()[0]:
            pygame.event.get()
            time.sleep(.0001)
        x, y = pygame.mouse.get_pos()
        choice = int(x / SPACING)
        return choice


def play_game(red_player: Player, blue_player: Player, delta=1, show=False, training=True, flip=.5, rand_start=False, rand_fuzz=0):
    global board, board_flipped

    # reset the board
    board = np.zeros((WIDTH, HEIGHT))
    board_flipped = False
    turn = 0

    # rand start can be used to create different game senarios
    if rand_start:
        for v in (-1, 1):
            place(v, rand.randint(0, 6), board)
        turn = 2

    # the history just keeps track of all the game states for training
    redHistory = np.zeros(shape=(0, WIDTH, HEIGHT, 1))
    blueHistory = np.zeros(shape=(0, WIDTH, HEIGHT, 1))

    winner = 0

    # play the game
    while winner == 0 and turn < WIDTH*HEIGHT:

        # board flip can also be used to mix things up
        if flip > rand.random():
            board = np.flip(board, axis=0)
            board_flipped = not board_flipped

        # for a model, 1's are always it's pieces, and -1's are always the enemies
        # as a result, for red, the sign's of the board must be flipped, so the RED board is the antiBoard
        antiBoard = -1*board

        redChoice = red_player.get_choice(antiBoard)
        succesfull, winner = place(-1, redChoice, board)

        redHistory = np.append(arr=redHistory, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
        # History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)

        if show:
            showGUI(surface)
            time.sleep(delta)

        turn += 1

        if winner == 0:

            if flip > rand.random():
                board = np.flip(board, axis=0)
                board_flipped = not board_flipped

            blueChoice = blue_player.get_choice(board)
            succesfull, winner = place(1, blueChoice, board)

            blueHistory = np.append(arr=blueHistory, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
            # History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)

            if show:
                showGUI(surface)
                time.sleep(delta)

            turn += 1

    if training:
        redHistory = -1*redHistory
        if winner == 1:  # blue is winner
            red_score = 0
            blue_score = 1
        if winner == -1:
            red_score = 1
            blue_score = -1
        if winner == 0:
            red_score = .5
            blue_score = .5

        if red_player.trainable:
            red_player.train(game_history=redHistory, score=red_score)
        if blue_player.trainable:
            blue_player.train(game_history=blueHistory, score=blue_score)

    time.sleep(.003)

    return winner, turn


def locate(lname: str, version: str):
    return 'saves/'+lname+'/'+lname+'_connect4_'+version+'.h5'


def give_report(trained_wins, control_wins, ties):
    growth_factor = trained_wins / control_wins
    CRITICAL_Z = 1.96  # 95% confidence
    marginal_error = CRITICAL_Z * math.sqrt(.25 / (trained_wins + control_wins))
    print('\nResults----------------------------------------')
    print('TOTAL SAMPLE GAMES: ' + str(trained_wins+control_wins+ties))
    print('\t| trained_wins\t| control_wins\t| ties')
    print('\t|\t' + str(trained_wins) + '\t\t\t|\t ' + str(control_wins) + '\t\t\t|\t ' + str(ties))
    print('IMPROVEMENT_FACTOR: ' + str(round(growth_factor, 3)))
    print('PERFORMANCE_CHANGE: ' + str(round((growth_factor - 1) * 100, 1)) + '%')
    print('SAMPLE_INDUCED_MAX_MARGINAL_ERROR: +/- ' + str(round(100 * marginal_error, 1)) + '%\n\t*at 95% confidence')
    print('-------------------------------------------------\n')

# ------------------------------------------------------------------------------------------------------ #

# RED: -1, BLUE: 1

MODEL_CONSTRUCTOR = more_dense_sigs_model

FIRST_TIME = True            # default: FALSE
SAVE = True                  # default: TRUE

CHOICE_CORRECT = True
RANDOM_START = True
SNAPSHOT_DELTA = 1000
EVAL_DELTA = 100

TRAIN_COUNT = 1001
PLAY_COUNT = 0
COMPETE_COUNT = 0

PLAY_SPEED = .4

NAME = 'HAL'

# make sure the name we want has a directory for it
if SAVE:
    if not os.path.exists('saves/'+NAME):
        os.makedirs('saves/'+NAME)

X_Snapshots = []
if FIRST_TIME:
    X_Model = MODEL_CONSTRUCTOR()  # make a new model with the set constructor
    X_Snapshots.append(MODEL_CONSTRUCTOR())
    if SAVE:
        X_Model.save(locate(NAME, 'MAIN'))
        X_Model.save(locate(NAME, str(len(X_Snapshots) - 1)))
else:
    X_Model = keras.models.load_model(locate(NAME, 'MAIN')) # if not, get the main model saved to memory
    total_snapshots = len(os.listdir('saves/' + NAME)) - 1  # need total snapshots
    for n in range(total_snapshots):
        snapshotModel = keras.models.load_model(locate(NAME, str(n)))
        X_Snapshots.append(snapshotModel)

X_Player = ModelPlayer(name=NAME,  model=X_Model, is_training=True, choice_correct=CHOICE_CORRECT)  # create a model player with the main model
User_Player = UserPlayer('User')

if TRAIN_COUNT > 0:
    snapped_wins = 0
    trained_wins = 0
    ties = 0
    win_record = []
    first_half_wins = 0
    second_half_wins = 0
    first_half_losses = 0
    second_half_losses = 0

    for n in range(TRAIN_COUNT):
        snapChoice = math.floor(rand.random()*len(X_Snapshots))
        SNAPPED_PLAYER = ModelPlayer(name=str(snapChoice), model=X_Snapshots[snapChoice], is_training=False, choice_correct=CHOICE_CORRECT)

        if rand.random() > .5:
            RED_PLAYER = X_Player
            BLUE_PLAYER = SNAPPED_PLAYER
        else:
            RED_PLAYER = SNAPPED_PLAYER
            BLUE_PLAYER = X_Player

        names = {-1: RED_PLAYER.name, 0: 'tie', 1: BLUE_PLAYER.name}

        gwinner, turns = play_game(RED_PLAYER, BLUE_PLAYER, 0, rand_start=RANDOM_START, training=True)
        print(str(n) + ':\t' + names[-1] + '\tvs.\t' + names[1]+'\t\t\tWinner: '+names[gwinner]+'\t\t\t'+'Turns: '+str(turns))

        if names[gwinner] == X_Player.name:
            trained_wins += 1
            win_record.append(1)
            if n % SNAPSHOT_DELTA / SNAPSHOT_DELTA < .5:
                first_half_wins += 1
            else:
                second_half_wins += 1
        if names[gwinner] == SNAPPED_PLAYER.name:
            snapped_wins += 1
            win_record.append(-1)
            if n % SNAPSHOT_DELTA / SNAPSHOT_DELTA < .5:
                first_half_losses += 1
            else:
                second_half_losses += 1
        if gwinner == 0:
            ties += 1
            win_record.append(0)

        if SAVE and n % 20 == 0:
            X_Model.save(locate(NAME, 'MAIN'))
            print('saving...')

        if n % SNAPSHOT_DELTA == 0 and n != 0 and False:
            X_Snapshots.append(keras.models.clone_model(X_Model))
            if SAVE:
                X_Model.save(locate(NAME, 'MAIN'))
                X_Model.save(locate(NAME, str(len(X_Snapshots)-1)))
                print('saving...')

        if n % EVAL_DELTA == 0 and n != 0:
            give_report(trained_wins, snapped_wins, ties)
            print(f'win/loss first half: {first_half_wins} / {first_half_losses}')
            print(f'win/loss second half: {second_half_wins} / {second_half_losses}')

            first_half_losses = 0
            first_half_wins = 0
            second_half_losses = 0
            second_half_wins = 0
            snapped_wins = 0
            trained_wins = 0
            ties = 0

            if (trained_wins > 90):
                print('YAY. We have over 90 wins! Time to stop now!')
                break


if SAVE:

    print('all models saved to memory')



