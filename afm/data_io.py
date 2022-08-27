import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import pickle
from .misc import get_line_point_coords, progress_bar
from .math_funcs import sma_shift
from igor.binarywave import load as load_
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_files(directory, req_ext=None):
    '''
    gets all the files in the given directory
    :param directory: str directory from which you want to load files from
    :param req_ext: optional str required tc_data extension
    :return: list of str names of the files in the given directory
    '''
    if req_ext is None:
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        return [os.path.join(directory, f) for f in os.listdir(directory) if
                os.path.isfile(os.path.join(directory, f)) and req_ext in f]


def get_folders(directory):
    '''
    gets all the folders in the given directory
    :param directory: str directory from which you want the sub-directories
    :return: list of str names of the sub-directories
    '''
    return [f.path for f in os.scandir(directory) if f.is_dir()]


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
    data = load_(filename)
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


def format_fd(df, k, pct_smooth=0.01):
    f = df.Defl.values * k
    h = (df.ZSnsr - df.Defl).values
    j = np.argmax(f[:int(0.8 * f.size)])
    win_size = int(pct_smooth * j)
    f = sma_shift(f[:j], win_size)
    i = np.argmin(f)
    return pd.DataFrame({'f': abs(f[i:] - f[i]), 'h': abs(sma_shift(h[i:j], win_size))})


class ForceMap:

    def __init__(self, root_directory, spring_const=None, sampling_frequency=None, probe_radius=1, contact_beta=3 / 2,
                 pct_smooth=0.01):
        self.root_directory = root_directory
        self.map_directory = None
        self.shape = None
        self.dimensions = None
        self.spring_const = spring_const
        self.sampling_frequency = sampling_frequency
        self.contact_alpha = 16 * np.sqrt(probe_radius) / 3
        self.contact_beta = contact_beta
        self.map_scalars = {}
        self.map_vectors = None
        self.fd_curves = None
        self.x = None
        self.y = None
        self.pct_smooth = pct_smooth
        self.feature_mask = None

        self.load_map()

    def load_map(self):
        # find files
        files = get_files(self.root_directory)
        # find the map file
        possible_map = [file for file in files if not all(kw in file for kw in ['Line', 'Point'])]
        if len(possible_map) != 1:
            exit('the .ibw file for the map height data is missing or duplicated')
        self.map_directory = possible_map[0]

        print('loading map data...', end='\r')
        map_dict = ibw2dict(self.map_directory)
        if self.spring_const is None:
            self.spring_const = map_dict['notes']['SpringConstant']
        self.dimensions = np.array([map_dict['notes']['ScanSize'], map_dict['notes']['ScanSize']])
        for i, label in enumerate(map_dict['labels']):
            data = np.array(map_dict['data'])[:, :, i]
            if i == 0:
                self.shape = data.shape
                x, y = np.linspace(0, self.dimensions[0], self.shape[0]), np.linspace(0, self.dimensions[1],
                                                                                      self.shape[1])
                self.x, self.y = np.meshgrid(x, y)
            self.map_scalars.update({label: data.T})
        print('done', end='\r')

        self.map_vectors = np.zeros(self.shape, dtype='object')
        self.fd_curves = np.zeros(self.shape, dtype='object')
        self.feature_mask = np.ones(self.shape) == 1
        for i, file in enumerate(files):
            progress_bar(i, len(files) - 1, message='loading force curves')
            if file == self.map_directory:
                continue
            coords = get_line_point_coords(file)
            self.map_vectors[coords] = ibw2df(file)
        print('done', end='\r')

    def transpose(self):
        for key, value in self.map_scalars.items():
            self.map_scalars[key] = value.T[::-1]
        self.map_vectors = self.map_vectors.T[::-1]
        self.fd_curves = self.fd_curves.T[::-1]

    def plot_map(self):
        figs, axs = plt.subplots(1, len(self.map_scalars.keys()))
        for i, (ax, key) in enumerate(zip(axs, self.map_scalars.keys())):
            a_i = ax.contourf(self.x, self.y, self.map_scalars[key])
            ax.set_title(key)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            figs.colorbar(a_i, cax=cax, orientation='vertical')
        plt.tight_layout()
        plt.show()

    # TODO make 3d surface plot

    def flatten_and_shift(self, order=1, left_right_mask=[0, 1], up_down_mask=[0, 1], show_plots=True):
        # l1 optimization of background shift to minimize outlier error
        def obj(X, func, real):
            return np.sum(abs(func(X) - real) ** 2)

        if order < 0:
            exit('flattening below order 0 doesnt make sense to me so i will crash now :)')
        # sp.optimize.minimize()
        if 'MapHeight' not in self.map_scalars.keys():
            exit('there is no height data')
        height = self.map_scalars['MapHeight'].copy()

        mask = np.ones(height.shape)
        mask[int(up_down_mask[0] * mask.shape[0]): int(up_down_mask[1] * mask.shape[0]),
        int(left_right_mask[0] * mask.shape[1]): int(left_right_mask[1] * mask.shape[1])] = 0

        if show_plots:
            plt.imshow(mask)
            plt.title('Mask')
            plt.show()

        if order == 0:
            height -= np.min(height)

        elif order == 1:
            def lin(X):
                A, B, C = X
                return A * self.x + B * self.y + C

            x_opt = minimize(obj, x0=[0, 0, 0], args=(lin, height), method='Nelder-Mead').x
            height -= lin(x_opt)

        elif order == 2:
            def quad(X):
                A, B, C, D, E = X
                return A * self.x ** 2 + B * self.x + C * self.y ** 2 + D * self.y + E

            x_opt = minimize(obj, x0=[0, 0, 0, 0, 0], args=(quad, height), method='Nelder-Mead').x
            height -= quad(x_opt)

        self.map_scalars.update({'MapFlattenHeight': height - np.min(height)})

    def format_fds(self):
        tot = 0
        for i, row in enumerate(self.map_vectors):
            for j, df in enumerate(row):
                tot += 1
                progress_bar(tot, self.shape[0] * self.shape[1], message='formatting force curves')
                self.fd_curves[i, j] = format_fd(df, self.spring_const, self.pct_smooth)
        print('done', end='\r')

    def cut_background(self, mult, show_plots=True):
        height_map = self.map_scalars['MapFlattenHeight'].copy()
        heights = height_map.ravel()
        cut = np.mean(heights) * mult
        self.feature_mask = height_map > cut
        if show_plots:
            figs, axs = plt.subplots(1, 2)
            axs[0].hist(heights, bins=100, label='Full')
            axs[0].hist(heights[heights > cut], bins=100, label='Cut')
            axs[0].legend()
            axs[0].set_title('Height Histogram')
            height_map[np.invert(self.feature_mask)] = -1
            masked_array = np.ma.masked_where(height_map == -1, height_map)
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='white')
            axs[1].contourf(self.x, self.y, masked_array, cmap=cmap)
            axs[1].set_xlabel('x (m)')
            axs[1].set_ylabel('y (m)')
            axs[1].set_title('Masked')
            plt.tight_layout()
            plt.show()

    # TODO thin sample correction

    # TODO tilted sample correction
