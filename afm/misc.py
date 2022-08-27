import re
import os

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

    print(message + f': [{arrow}{padding}] {int(fraction*100)}%', end=ending)

