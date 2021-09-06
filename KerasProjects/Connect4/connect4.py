import time
import pygame
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import random as rand
import math
import os

TEMPERATURE = 80

WIDTH = 7
HEIGHT = 6
SPACING = 60
RAD = 20
COLORS = {
    -1: (255, 0, 0),
    0: (255, 255, 255),
    1: (0, 0, 255)
}

board_flipped = False
surface = None
board = np.zeros((WIDTH, HEIGHT))


def setupGUI():
    pygame.init()
    surface = pygame.display.set_mode((SPACING * WIDTH, SPACING * HEIGHT))
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
                (int((x + .5) * SPACING), int((HEIGHT - y - .5) * SPACING)),
                RAD
            )
    pygame.display.flip()


def test_win(x, y, my_board=board, side=None):
    if side is None:
        main = my_board.item((x, y))
    else:
        main = side
    dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
    for direc in dirs:
        count = 1
        for sign in (-1, 1):
            tx = x + sign * direc[0]
            ty = y + sign * direc[1]
            stop = False
            while 0 <= tx < WIDTH and 0 <= ty < HEIGHT and not stop:
                if my_board.item((tx, ty)) == main:
                    count += 1
                    tx += sign * direc[0]
                    ty += sign * direc[1]
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

    return successes, this_predictions.reshape(7)


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


def train_game(history: np.ndarray, model: keras.Model, score=1, decay_rate = .99):
    size = history.shape[0]

    train_for = np.zeros(shape=size)
    for n in range(size):
        train_for.itemset(size - n - 1, (score - .5) * math.pow(decay_rate, n) + .5)

    # train_for.fill(score)
    model.fit(history, train_for, verbose=0)


class Player:
    def __init__(self, name, trainable):
        self.trainable = trainable
        self.name = name

    def get_choice(self, game_board, isProportionalChoice=False, temperature=TEMPERATURE):
        pass

    def train(self, game_history, score):  # x is the input group, score
        pass


class ModelPlayer(Player):
    def __init__(self, name, model, is_training, choice_correct, save_location=None):
        super().__init__(name, is_training)
        self.model = model
        self.choice_correct = choice_correct
        self.save_location = save_location

    def get_choice(self, eval_board, isProportionalChoice=False, temperature=TEMPERATURE):
        sucesses, values = evaluate_actions(eval_board, self.model)  # get a list of confidences in choice
        #print(values)


        if isProportionalChoice:
            probs = np.exp(temperature * values)
            for i in range(7):
                if not sucesses[i]:
                    probs[i] = 0
            probs = probs / np.sum(probs)
            #print(probs)
            choice = np.random.choice(np.arange(0,7),p=probs) # pick a random
        else:
            for i in range(7):
                if not sucesses[i]:
                    values[i] = 0
            choice = np.argmax(values)  # pick the one with the strongest confidence


        if self.choice_correct:  # if choice correct is on, automatically fix errors
            choice = correct_choice(choice, eval_board)
        return choice

    def train(self, game_history, score):  # x is the input group, score
        if self.trainable:
            train_game(history=game_history, model=self.model, score=score)


class UserPlayer(Player):
    def __init__(self, name):
        super().__init__(name=name, trainable=False)

    def get_choice(self, eval_board, isProportionalChoice=False, temperature=0):
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


def play_game(red_player: Player, blue_player: Player, delta=0, show=False, training=True, flip=.5, rand_start=False,
              proportional_choice=False):
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
    while winner == 0 and turn < WIDTH * HEIGHT:

        # board flip can also be used to mix things up
        if flip > rand.random():
            board = np.flip(board, axis=0)
            board_flipped = not board_flipped

        # for a model, 1's are always it's pieces, and -1's are always the enemies
        # as a result, for red, the sign's of the board must be flipped, so the RED board is the antiBoard
        antiBoard = -1 * board

        redChoice = red_player.get_choice(antiBoard, isProportionalChoice=proportional_choice)
        succesfull, winner = place(-1, redChoice, board)

        redHistory = np.append(arr=redHistory, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
        # History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)

        if red_player.__class__ != UserPlayer:
            time.sleep(delta)
        if show:
            showGUI(surface)

        turn += 1

        if winner == 0:

            if flip > rand.random():
                board = np.flip(board, axis=0)
                board_flipped = not board_flipped

            blueChoice = blue_player.get_choice(board, isProportionalChoice=proportional_choice)
            succesfull, winner = place(1, blueChoice, board)

            blueHistory = np.append(arr=blueHistory, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)
            # History = np.append(arr=History, values=board.reshape([1, WIDTH, HEIGHT, 1]), axis=0)

            if blue_player.__class__ != UserPlayer:
                time.sleep(delta)
            if show:
                showGUI(surface)

            turn += 1

    if training:
        redHistory = -1 * redHistory
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


def locate(lname: str, version: str, extension=''):
    return 'saves/' + lname + '/' + lname + '_connect4_' + version + '.' + extension


def give_report(trained_wins, control_wins, ties):
    # growth_factor = trained_wins / control_wins
    # CRITICAL_Z = 1.96  # 95% confidence
    # marginal_error = CRITICAL_Z * math.sqrt(.25 / (trained_wins + control_wins))
    print('\nResults------------------------------------------')
    print('TOTAL SAMPLE GAMES: ' + str(trained_wins + control_wins + ties))
    print('\t| trained_wins\t| control_wins\t| ties')
    print('\t|\t' + str(trained_wins) + '\t\t\t|\t ' + str(control_wins) + '\t\t\t|\t ' + str(ties))
    # print('WIN_FACTOR: ' + str(round(growth_factor, 3)))
    # print('PERFORMANCE_CHANGE: ' + str(round((growth_factor - 1) * 100, 1)) + '%')
    # print('SAMPLE_INDUCED_MAX_MARGINAL_ERROR: +/- ' + str(round(100 * marginal_error, 1)) + '%\n\t*at 95% confidence')
    print('-------------------------------------------------\n')


# ------------------------------------------------------------------------------------------------------ #

# model constructors

def more_dense_sigs_model(learning_rate=.00005):
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.BinaryCrossentropy())
    # model.build(input_shape=(WIDTH+6, HEIGHT, 2))
    print(model.summary())
    # print(model.layers[1].get_weights())
    return model


def BIG_model(learning_rate=.00005):
    model = keras.Sequential([
        keras.layers.ZeroPadding2D(padding=2, input_shape=(WIDTH, HEIGHT, 1)),
        keras.layers.Conv2D(50, (5, 5), activation='sigmoid'),
        keras.layers.ZeroPadding2D(padding=2),
        keras.layers.Conv2D(200, (5, 5), activation='sigmoid'),
        keras.layers.ZeroPadding2D(padding=1),
        keras.layers.Conv2D(50, (4, 4), activation='sigmoid'),
        keras.layers.Flatten(),
        keras.layers.Dense(200, activation='sigmoid'),
        keras.layers.Dense(60, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.BinaryCrossentropy())
    # model.build(input_shape=(WIDTH+6, HEIGHT, 2))
    print(model.summary())
    # print(model.layers[1].get_weights())
    return model


def BIG_relu_model(learning_rate=.00005):
    model = keras.Sequential([
        keras.layers.ZeroPadding2D(padding=2, input_shape=(WIDTH, HEIGHT, 1)),
        keras.layers.Conv2D(50, (5, 5), activation='relu'),
        keras.layers.ZeroPadding2D(padding=2),
        keras.layers.Conv2D(200, (5, 5), activation='relu'),
        keras.layers.ZeroPadding2D(padding=1),
        keras.layers.Conv2D(50, (4, 4), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.BinaryCrossentropy())
    # model.build(input_shape=(WIDTH+6, HEIGHT, 2))
    print(model.summary())
    # print(model.layers[1].get_weights())
    return model

def BIG_BIG_BIG_model(learning_rate=.000005):
    model = keras.Sequential([
        keras.layers.ZeroPadding2D(padding=2, input_shape=(WIDTH, HEIGHT, 1)),
        keras.layers.Conv2D(50, (5, 5), activation='relu'),
        keras.layers.ZeroPadding2D(padding=2),
        keras.layers.Conv2D(200, (5, 5), activation='relu'),
        keras.layers.ZeroPadding2D(padding=1),
        keras.layers.Conv2D(200, (5, 5), activation='relu'),
        keras.layers.ZeroPadding2D(padding=1),
        keras.layers.Conv2D(200, (5, 5), activation='relu'),
        keras.layers.ZeroPadding2D(padding=1),
        keras.layers.Conv2D(50, (4, 4), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.BinaryCrossentropy())
    # model.build(input_shape=(WIDTH+6, HEIGHT, 2))
    print(model.summary())
    # print(model.layers[1].get_weights())
    return model


class ResnetModel(keras.Model):

    def __init__(self, resLayers=10, learning_rate=.00001):
        super(ResnetModel, self).__init__()

        self.initPad = keras.layers.ZeroPadding2D(padding=2)
        self.initConvolve = keras.layers.Conv2D(20, (5, 5), activation='relu')

        self.convolves = []
        for n in range(1, resLayers):
            self.convolves.append([
                keras.layers.ZeroPadding2D(padding=2),   #, input_shape=(WIDTH, HEIGHT, 1)),
                keras.layers.Conv2D(20, (5, 5), activation='relu'),
                keras.layers.ZeroPadding2D(padding=2),
                keras.layers.Conv2D(20, (5, 5), activation='relu')
            ])


        self.linear = [
            keras.layers.Flatten(),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dense(200, activation='relu'),
            keras.layers.Dense(60, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ]

        self.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=keras.losses.BinaryCrossentropy())
        self.fit(tf.ones(shape=(1, WIDTH, HEIGHT, 1)),tf.ones(shape=1))

        #print(self.summary())


    def call(self, x):
        #print(x.shape)
        x = self.initPad(x)
        #print(x.shape)
        x = self.initConvolve(x)
        #print(x.shape)

        for res in self.convolves:
            temp = x
            for layer in res:
                x = layer(x)
                #print(x.shape)
            x = x + temp

        for layer in self.linear:
            x = layer(x)
            #print(x.shape)

        return x


MODEL_CONSTRUCTOR = ResnetModel


# ------------------------------------------------------------------------------------------------------ #

# training and simulation methods

def snapshot_train(player_class_name, n_games=1000, snap_delta=200, eval_delta=100, random_start=False,
                   save=True, flip=.4, choice_correct=True, benchmark_player=None, n_eval=20):
    snapped_wins = 0
    trained_wins = 0
    ties = 0
    snapshots = []

    # if first_time:
    #     snapshots.append(model_constructor())
    #     main_model = model_constructor()
    #     if save:
    #         if not os.path.exists('saves/' + player_class_name):
    #             os.makedirs('saves/' + player_class_name)
    #         main_model.save(locate(player_class_name, 'MAIN'))
    #         main_model.save(locate(player_class_name, str(len(snapshots) - 1)))
    # else:

    main_model = keras.models.load_model(locate(player_class_name, 'MAIN'))
    total_snapshots = len(os.listdir('saves/' + player_class_name)) - 1
    for n in range(total_snapshots):
        snapshotModel = keras.models.load_model(locate(player_class_name, str(n)))
        snapshots.append(snapshotModel)

    main_player = ModelPlayer(name=player_class_name, model=main_model, is_training=True, choice_correct=choice_correct)

    for n in range(n_games):
        snap_choice = math.floor(math.pow(rand.random(), .4) * len(snapshots))
        snapped_player = ModelPlayer(name=str(snap_choice), model=snapshots[snap_choice], is_training=False,
                                     choice_correct=choice_correct)

        if rand.random() > .5:
            RED_PLAYER = main_player
            BLUE_PLAYER = snapped_player
        else:
            RED_PLAYER = snapped_player
            BLUE_PLAYER = main_player

        names = {-1: RED_PLAYER.name, 0: 'tie', 1: BLUE_PLAYER.name}

        gwinner, turns = play_game(RED_PLAYER, BLUE_PLAYER, rand_start=random_start, training=True, flip=flip)
        print(f'{str(n)}:\t{names[-1]}\tvs.\t{names[1]}\t\t\tWinner: {names[gwinner]}\t\t\tTurns: {str(turns)}')

        if names[gwinner] == main_player.name:
            trained_wins += 1
        if names[gwinner] == snapped_player.name:
            snapped_wins += 1
        if gwinner == 0:
            ties += 1

        # if save and n % 20 == 0:
        #     main_model.save(locate(player_class_name, 'MAIN'))
        #     print('saving...')

        if n % snap_delta == 0 and n != 0:
            if save:
                main_model.save(locate(player_class_name, 'MAIN'))
                main_model.save(locate(player_class_name, str(len(snapshots))))
                print('saving...')
            snapshots.append(keras.models.load_model(locate(player_class_name, 'MAIN')))
            print('added snapshot version '+str(len(snapshots)-1))

        if n % eval_delta == 0 and n != 0:
            give_report(trained_wins, snapped_wins, ties)

            if benchmark_player is not None and n_eval > 0:
                control_wins = 0
                trained_wins = 0
                ties = 0
                for eval_i in range(n_eval):
                    if rand.random() > .5:
                        RED_PLAYER = main_player
                        BLUE_PLAYER = benchmark_player
                    else:
                        RED_PLAYER = benchmark_player
                        BLUE_PLAYER = main_player

                    names = {-1: RED_PLAYER.name, 0: 'tie ', 1: BLUE_PLAYER.name}

                    gwinner, turns = play_game(RED_PLAYER, BLUE_PLAYER, training=False, flip=flip)
                    print(
                        f'EVAL {str(eval_i)}:\t{names[-1]}   vs.   {names[1]}\t\t\tWinner: {names[gwinner]}\t\t\tTurns: {str(turns)}')

                    if gwinner == 0:
                        ties += 1
                    else:
                        if names[gwinner] == benchmark_player.name:
                            control_wins += 1
                        else:
                            trained_wins += 1

                give_report(trained_wins, control_wins, ties)

            snapped_wins = 0
            trained_wins = 0
            ties = 0


def setup_snapshot_train_folder(player_class_name, model, extension='h5'):
    main_model = model
    if not os.path.exists('saves/' + player_class_name):
        os.makedirs('saves/' + player_class_name)
    main_model.save(locate(player_class_name, 'MAIN', extension=extension))
    main_model.save(locate(player_class_name, str(0), extension=extension))

    print(f'the setup for snapshot train folder {player_class_name} is complete')


################

def compete_train(folder: str = None, competing_players: list = None, benchmark_player: Player = None, n_rounds=10,
                  n_train=200, n_eval=30, training=True, save=True, flip=.4, choice_correct=True, pre_eval=True,
                  rand_start=True, proportional_choice=False):
    if competing_players is None:
        competing_players = []
        for filename in os.listdir('saves/' + folder):
            filepath = f'saves/{folder}/{filename}'
            print(filepath)
            model = keras.models.load_model(filepath)
            player = ModelPlayer(name=filename[:-3], model=model, is_training=True,
                                 choice_correct=choice_correct, save_location=filepath)
            competing_players.append(player)

    if benchmark_player is not None and pre_eval and n_eval > 0:
        benchmark_player.trainable = False
        control_wins = 0
        trained_wins = 0
        ties = 0
        for eval_i in range(n_eval):
            if rand.random() > .5:
                RED_PLAYER = rand.choice(competing_players)
                BLUE_PLAYER = benchmark_player
            else:
                RED_PLAYER = benchmark_player
                BLUE_PLAYER = rand.choice(competing_players)

            names = {-1: RED_PLAYER.name, 0: 'tie ', 1: BLUE_PLAYER.name}

            gwinner, turns = play_game(RED_PLAYER, BLUE_PLAYER, training=False, flip=flip)
            print(
                f'PRE-EVAL {str(eval_i)}:\t{names[-1]}   vs.   {names[1]}\t\t\tWinner: {names[gwinner]}\t\t\tTurns: {str(turns)}')

            if gwinner == 0:
                ties += 1
            else:
                if names[gwinner] == benchmark_player.name:
                    control_wins += 1
                else:
                    trained_wins += 1

        give_report(trained_wins, control_wins, ties)

    n = 0
    for rounds in range(n_rounds):
        for train in range(n_train):
            n = n + 1
            RED_PLAYER = rand.choice(competing_players)
            BLUE_PLAYER = rand.choice(competing_players)
            names = {-1: RED_PLAYER.name, 0: 'tie', 1: BLUE_PLAYER.name}

            gwinner, turns = play_game(RED_PLAYER, BLUE_PLAYER, training=training, rand_start=rand_start, proportional_choice=proportional_choice)
            print(f'{str(n)}:\t{names[-1]}\tvs.\t{names[1]}\t\t\tWinner: {names[gwinner]}\t\t\tTurns: {str(turns)}')

        if save:
            for player in competing_players:
                player.model.save(player.save_location)
            print('saving...')

        if benchmark_player is not None and n_eval > 0:
            control_wins = 0
            trained_wins = 0
            ties = 0
            for eval_i in range(n_eval):
                if rand.random() > .5:
                    RED_PLAYER = rand.choice(competing_players)
                    BLUE_PLAYER = benchmark_player
                else:
                    RED_PLAYER = benchmark_player
                    BLUE_PLAYER = rand.choice(competing_players)

                names = {-1: RED_PLAYER.name, 0: 'tie ', 1: BLUE_PLAYER.name}

                gwinner, turns = play_game(RED_PLAYER, BLUE_PLAYER, training=False, flip=flip)
                print(
                    f'EVAL {str(eval_i)}:\t{names[-1]}   vs.   {names[1]}\t\t\tWinner: {names[gwinner]}\t\t\tTurns: {str(turns)}')

                if gwinner == 0:
                    ties += 1
                else:
                    if names[gwinner] == benchmark_player.name:
                        control_wins += 1
                    else:
                        trained_wins += 1

            give_report(trained_wins, control_wins, ties)


def setup_compete_group_folder(group_name, named_models, extension='h5'):
    if not os.path.exists('saves/' + group_name):
        os.makedirs('saves/' + group_name)
    if extension=='h5' or True:
        for name, model in named_models.items():
            model.save(f'saves/{group_name}/{name}.{extension}')
    # else:
    #     for name, model in named_models.items():
    #         model.save_weights(f'saves/{group_name}/{name}.{extension}')
    print(f'the setup for competing group folder {group_name} is complete')


##################


def play_user(ai_player: ModelPlayer, user_player: UserPlayer = UserPlayer('you'), n_games=5, play_speed=.5):
    for n in range(n_games):
        if rand.random() > .5:
            RED_PLAYER = user_player
            BLUE_PLAYER = ai_player
        else:
            RED_PLAYER = user_player
            BLUE_PLAYER = ai_player
        names = {-1: RED_PLAYER.name, 0: 'tie', 1: BLUE_PLAYER.name}
        print(str(n) + ':\t' + names[-1] + '\tvs.\t' + names[1])
        gwinner, turns = play_game(RED_PLAYER, BLUE_PLAYER, delta=play_speed, rand_start=False,
                                   training=False, show=True, flip=0)
        print('\t\t\tWinner: ' + names[gwinner] + '\t\t\t' + 'Turns: ' + str(turns))


def play_self():
    pass


# ------------------------------------------------------------------------------------------------------ #

# main loop

# RED: -1, BLUE: 1

if __name__ == '__main__':

    # testModel = ResnetModel()
    # testModel.summary()
    # play_user(ModelPlayer(model=testModel, name='testModel', is_training=False, choice_correct=True), n_games=1)


    groupName = 'RandRes10'

    is_play = False
    is_new = False

    if is_new:
        setup_compete_group_folder(groupName,
                                   {
                                       "Adam": ResnetModel(resLayers=10),
                                       "Jane": ResnetModel(resLayers=10),
                                       "Mark": ResnetModel(resLayers=10)
                                   },
                                   extension='')

        # setup_snapshot_train_folder('resHAL', ResnetModel(resLayers=20), extension='')

    if not is_play:
        bench = ModelPlayer(model=keras.models.load_model(locate('HAL', 'MAIN', extension='h5')), name='HAL', is_training=False,
                            choice_correct=True)

        compete_train(groupName, benchmark_player=bench, n_rounds=40, rand_start=False, proportional_choice=True)

        #snapshot_train('ResHAL', n_games=4000, benchmark_player=bench)

    else:
        #bench = ModelPlayer(model=keras.models.load_model(locate('HAL', 'MAIN', extension='h5')), name='HAL',
        #                    is_training=False, choice_correct=True)

        adam = ModelPlayer('Adam', keras.models.load_model(f'saves/{groupName}/Adam'), is_training=False, choice_correct=True)
        jane = ModelPlayer('Jane', keras.models.load_model(f'saves/{groupName}/jane'), is_training=False, choice_correct=True)
        mark = ModelPlayer('Mark', keras.models.load_model(f'saves/{groupName}/Mark'), is_training=False, choice_correct=True)

        #play_user(bench, n_games=1)
        play_user(adam, n_games=2)
        play_user(jane, n_games=2)
        play_user(mark, n_games=2)

        # for n in (5,10,15,20):
        #     player = ModelPlayer(name=str(n), model=keras.models.load_model(locate('ResHAL', str(n), extension='')),
        #                                                          is_training=False, choice_correct=True)
        #
        #     play_user(player, n_games=3)

print('finished')
