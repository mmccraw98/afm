import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
import os
import pickle
from .misc import get_line_point_coords, progress_bar, get_window_size, get_elbow_min
from .math_funcs import sma_shift
from igor.binarywave import load as load_
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from mlinsights.mlmodel import KMeansL1L2


def next_path(path_pattern):
    """
    https://stackoverflow.com/questions/17984809/how-do-i-create-an-incrementing-filename-in-python
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


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


def format_fd(df, k, pct_smooth=0.01, num_attempts=10, n_pi_rot=0):
    f = df.Defl.values * k
    h = (df.ZSnsr - df.Defl).values
    j0 = np.argmax(df.ZSnsr.values)
    j = np.argmax(f[: j0])
    i0 = get_window_size(0.1, f.size)
    offset = get_window_size(0.01, j - i0)
    f_temp = sma_shift(f[i0: j], offset)
    i_vals = [0]
    for n in range(num_attempts):
        i_vals.append(np.argmin(f_temp[i_vals[-1]:]) + i_vals[-1])
    i = int(np.mean(i_vals[1:])) + offset
    i += np.argmin(f[i:j])
    win_size = get_window_size(pct_smooth, j - i)
    f = sma_shift(f[i:j], win_size)
    h = sma_shift(h[i:j], win_size)
    i = get_elbow_min(h, f, np.pi * n_pi_rot)
    f = f[i:] - f[i]
    h = h[i:] - h[i]
    return pd.DataFrame({'f': abs(f), 'h': abs(h)})


def merge_fmap_masks(maps):
    return np.prod([m.feature_mask for m in maps], axis=0)


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
        self.feature_mask = self.feature_mask.T[::-1]
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
        def obj(X, func, real, mask):
            return np.sum(abs(func(X) - real) * mask ** 2)

        if order not in [0, 1, 2]:
            exit('flattening {} doesnt make sense to me so i will crash now :)'.format(order))
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
            height -= np.min(height[mask == 1])

        elif order == 1:
            def lin(X):
                A, B, C = X
                return A * self.x + B * self.y + C

            x_opt = minimize(obj, x0=[0, 0, 0], args=(lin, height, mask), method='Nelder-Mead').x
            height -= lin(x_opt)

        elif order == 2:
            def quad(X):
                A, B, C, D, E = X
                return A * self.x ** 2 + B * self.x + C * self.y ** 2 + D * self.y + E

            x_opt = minimize(obj, x0=[0, 0, 0, 0, 0], args=(quad, height, mask), method='Nelder-Mead').x
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

    def ml_flatten_and_shift(self, num_features=2, order=1, show_plots=False):
        '''
        kind of experimental function for flattening the height of force maps and shifting them to start at 0 height
        there are assumed to be a certain number features in an image for instance, a cell sitting atop a culture dish
        gives two features: the cell and the dish.  we can then identify these features by their distinct heights
        (we use 2 by default) and then we take the lowest height out of the group and make a mask
        using the mask, we fit a surface of a given order to the mask and then subtract the fitted surface from the
        height data
        :param num_features: number of distinct topographical features in the force map
        :param order: order of surface plane fit for background subtraction
        :param show_plots: whether or not to show the mask image
        :return: adds a mapflattenheight element to self.map_scalars associated with the corrected height map
        and adds a feature_map to self corresponding to the non-cut portion of the height map
        '''

        # l1 optimization of background shift to minimize outlier error
        def obj(X, func, real, mask):
            return np.sum(abs(func(X) - real) * mask ** 2)

        if order not in [0, 1, 2]:
            exit('flattening {} doesnt make sense to me so i will crash now :)'.format(order))
        if 'MapHeight' not in self.map_scalars.keys():
            exit('there is no height data')

        self.format_fds()
        sizes = np.array([df.f.size for df in self.fd_curves.ravel()])
        height = self.map_scalars['MapHeight'].copy().ravel()
        model = KMeansL1L2(n_clusters=num_features, norm='L1', init='k-means++', random_state=42)
        data = np.concatenate([feature.reshape(-1, 1) for feature in [height, sizes]], axis=1)
        model.fit(data)
        background_label = np.argmin([np.mean(height[model.labels_ == label]) for label in np.unique(model.labels_)])
        mask = np.invert(model.labels_ == background_label).reshape(self.shape)
        height = height.reshape(self.shape)
        if show_plots:
            plt.imshow(mask)
            plt.title('Mask')
            plt.show()
        if order == 0:
            height -= np.min(height[np.invert(mask)])

        elif order == 1:
            def lin(X):
                A, B, C = X
                return A * self.x + B * self.y + C

            x_opt = minimize(obj, x0=[0, 0, 0], args=(lin, height, np.invert(mask)), method='Nelder-Mead').x
            height -= lin(x_opt)

        elif order == 2:
            def quad(X):
                A, B, C, D, E = X
                return A * self.x ** 2 + B * self.x + C * self.y ** 2 + D * self.y + E

            x_opt = minimize(obj, x0=[0, 0, 0, 0, 0], args=(quad, height, np.invert(mask)), method='Nelder-Mead').x
            height -= quad(x_opt)

        self.map_scalars.update({'MapFlattenHeight': height - np.min(height)})
        self.feature_mask = mask

    def copy(self):
        return deepcopy(self)

    # TODO thin sample correction

    # TODO tilted sample correction


def offset_polynom(x, y_offset, x_offset, slope):
    # model of an offset polynomial with power 2 (3/2 doesn't seem to work!)
    poly = 2
    d = x - x_offset
    return slope * (abs(d) - d) ** poly + y_offset

def fit_contact(df, k):
    # fit QUADRATIC f-z model to data and return contact location and index
    # format f-d dataframe values and cut to maximum
    f = df.Defl.values * k
    i = np.argmax(f)
    z_tip = (df.Defl - df.ZSnsr).values[:i]
    f = f[:i]

    # normalize f-d values
    x = z_tip - z_tip[0]
    x /= max(abs(x))
    y = f - f[0]
    y /= max(abs(f))

    # fit to the offset polynomial model
    X, _ = curve_fit(offset_polynom, x, y)
    y_offset, x_offset, slope = X

    # reverse the normalization and get index
    z_offset = x_offset * max(abs(z_tip - z_tip[0])) + z_tip[0]
    j = np.argmin((z_tip - z_offset) ** 2)
    return z_offset, j
