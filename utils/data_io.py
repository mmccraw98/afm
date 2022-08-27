import numpy as np
import pandas as pd
import os
import pickle
import re
import igor
from utils.misc import search_between

def get_files(dir, req_ext=None):
    '''
    gets all the files in the given directory
    :param dir: str directory from which you want to load files from
    :param req_ext: optional str required tc_data extension
    :return: list of str names of the files in the given directory
    '''
    if req_ext is None:
        return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    else:
        return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and req_ext in f]


def get_folders(dir):
    '''
    gets all the folders in the given directory
    :param dir: str directory from which you want the sub-directories
    :return: list of str names of the sub-directories
    '''
    return [f.path for f in os.scandir(dir) if f.is_dir()]


# appropriating some functions from from https://github.com/N-Parsons/ibw-extractor
def from_repr(s):
    """Get an int or float from its representation as a string"""
    # Strip any outside whitespace
    s = s.strip()
    # "NaN" and "inf" can be converted to floats, but we don't want this
    # because it breaks in Mathematica!
    if s[1:].isalpha():  # [1:] removes any sign
        rep = s
    else:
        try:
            rep = int(s)
        except ValueError:
            try:
                rep = float(s)
            except ValueError:
                rep = s
    return rep


def fill_blanks(lst):
    """Convert a list (or tuple) to a 2 element tuple"""
    try:
        return (lst[0], from_repr(lst[1]))
    except IndexError:
        return (lst[0], "")


def flatten(lst):
    """Completely flatten an arbitrarily-deep list"""
    return list(_flatten(lst))


def _flatten(lst):
    """Generator for flattening arbitrarily-deep lists"""
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        elif item not in (None, "", b''):
            yield item


def process_notes(notes):
    """Splits a byte string into an dict"""
    # Decode to UTF-8, split at carriage-return, and strip whitespace
    note_list = list(map(str.strip, notes.decode(errors='ignore').split("\r")))
    note_dict = dict(map(fill_blanks, [p.split(":") for p in note_list]))

    # Remove the empty string key if it exists
    try:
        del note_dict[""]
    except KeyError:
        pass
    return note_dict


def ibw2dict(filename):
    """Extract the contents of an *ibw to a dict"""
    data = igor.binarywave.load(filename)
    wave = data['wave']

    # Get the labels and tidy them up into a list
    labels = list(map(bytes.decode,
                      flatten(wave['labels'])))

    # Get the notes and process them into a dict
    notes = process_notes(wave['note'])

    # Get the data numpy array and convert to a simple list
    wData = np.nan_to_num(wave['wData']).tolist()

    # Get the filename from the file - warn if it differs
    fname = wave['wave_header']['bname'].decode()
    input_fname = os.path.splitext(os.path.basename(filename))[0]
    if input_fname != fname:
        print("Warning: stored filename differs from input file name")
        print("Input filename: {}".format(input_fname))
        print("Stored filename: {}".format(str(fname) + " (.ibw)"))

    return {"filename": fname, "labels": labels, "notes": notes, "data": wData}


def ibw2df(filename):
    data = ibw2dict(filename)
    headers = data['labels']
    return pd.DataFrame(data['data'], columns=headers)


def load(path, required_extension=None):
    '''
    loads data from a number of formats into python
    :param path: str path to thing being loaded in
    :param required_extension: str required extension for the file(s) to be loaded
    i.e. only load files with the required_extension
    :return: the data
    '''
    if not os.path.isfile(path):
        exit('invalid path')
    file_name = os.path.basename(path)  # tc_data name
    extension = file_name.split(sep='.')[-1]
    if extension == 'csv' or required_extension == 'csv':
        data = pd.read_csv(path)
    elif extension == 'xlsx' or required_extension == 'xlsx':
        data = pd.read_excel(path, engine='openpyxl')
    elif extension == 'txt' or required_extension == 'txt':
        with open(path, 'r') as f:
            data = f.read()
    elif extension == 'pkl' or required_extension == 'pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif extension == 'ibw' or required_extension == 'ibw':
        data = ibw2df(path)
    else:
        exit('extension not yet supported: {}'.format(file_name))
    return data


class force_map:

    def __init__(self, directory):
        self.directory = directory
        self.shape = None
        self.load_map()

    def load_map(self):
        print(self.directory)