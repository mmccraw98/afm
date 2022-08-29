import re
import os
import numpy as np


def get_window_size(percent, size):
    win_size = int(percent * size)
    return win_size + int(win_size == 0)


def search_between(start, end, string):
    return re.search('%s(.*)%s' % (start, end), string).group(1)


def get_line_point_coords(filename):
    just_the_name = os.path.split(filename)[-1].split('.')[0]
    line, point = just_the_name.strip('Line').split('Point')
    return int(line), int(point)


def progress_bar(current, total, bar_length=20, message='Loading'):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(message + f': [{arrow}{padding}] {int(fraction * 100)}%', end=ending)


def selectyesno(prompt):
    '''
    given a prompt with a yes / no input answer, return the boolean value of the given answer
    :param prompt: str a prompy with a yes / no answer
    :return: bool truth value of the given answer: yes -> True, no -> False
    '''
    print(prompt)  # print the user defined yes / no question prompt
    # list of understood yes inputs, and a list of understood no inputs
    yes_choices, no_choices = ['yes', 'ye', 'ya', 'y', 'yay'], ['no', 'na', 'n', 'nay']
    # use assignment expression to ask for inputs until an understood input is given
    while (choice := input('enter: (y / n) ').lower()) not in yes_choices + no_choices:
        print('input not understood: {} '.format(choice))
    # if the understood input is a no, it returns false, if it is a yes, it returns true
    return choice in yes_choices


def rotate_vector(data, angle):
    # make rotation matrix
    theta = np.radians(angle)
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))
    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)
    # return index of elbow
    return rotated_vector


def scale(arr):
    return (arr - min(arr)) / (max(arr) - min(arr))


def get_elbow_min(arr_x, arr_y, angle):
    data = rotate_vector(np.concatenate((scale(arr_x).reshape(-1, 1), scale(arr_y).reshape(-1, 1)), axis=1), angle)
    return np.argmin(data[:, 1])
