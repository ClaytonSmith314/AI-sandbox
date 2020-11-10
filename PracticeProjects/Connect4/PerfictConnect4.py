import numpy as np
import connect4 as c4
import json

FILE_PATH = 'saves/PerfectPlayer/data'

TOTAL_SOLVED = 0
STACK_HISTORY = []
LEVEL = 0
PRINT_LEVEL = 15
LEVEL_LIMIT = 20

def solve_connect4_puzzle(mapping: dict, board=np.zeros(shape=(c4.WIDTH, c4.HEIGHT)), competence=.95):
    global TOTAL_SOLVED, STACK_HISTORY, LEVEL

    LEVEL += 1


    myId = make_board_id(board)
    if False and myId in mapping.values():
        PreventScore = mapping[myId][1]
    else:
        count = 0
        sumRating = 0
        choice = 0
        maxRating = -1
        winningPos = False
        for n in range(7):

            if LEVEL <= PRINT_LEVEL:
                STACK_HISTORY.append(n)
                print_stack()


            newBoard = board.copy() * -1
            succesfull, win = c4.place(1, n, newBoard)
            if succesfull:
                if win:
                    pathRating = 1
                    winningPos = True
                    choice = n
                else:
                    if LEVEL > LEVEL_LIMIT:
                        pathRating = 0
                    else:
                        pathRating = solve_connect4_puzzle(mapping, newBoard)
                sumRating += pathRating
                count += 1
                if pathRating > maxRating:  # change to a new rating if it's max rating
                    choice = n
                    maxRating = pathRating

            if LEVEL <= PRINT_LEVEL:
                STACK_HISTORY.pop()
                print_stack()


        if winningPos:  # if it is a win, preventation is negative
            PreventScore = -1
        else:
            if count == 0:  # if it is a tie, preventation is neutral
                PreventScore = 0
            else:
                # the preventing score is the score if the opposing player picked the correct choice 95% of the time
                # and a random choice 5% of the time
                PreventScore = -(competence*maxRating + (1-competence) * sumRating / count)

        mapping[myId] = (choice, PreventScore)

        LEVEL -= 1

    return PreventScore


def make_board_id(board):
    idstr = ''
    for group in range(3):
        number = 0
        for i in range(2 * 7):
            number += (board[i % 7, i // 7 + 2 * group] + 1) * pow(3, i)
        idstr = idstr + str(number)
    return idstr


def print_stack():
    global STACK_HISTORY
    string = ' : '
    for n in STACK_HISTORY:
        string = string + str(n) + '> '
    print(string)


if __name__ == '__main__':
    myMapping = {}
    print('solving...')
    solve_connect4_puzzle(myMapping)
    print('completed!')
    json.dump(myMapping, FILE_PATH, indent=1)





