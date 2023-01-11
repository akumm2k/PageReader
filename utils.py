from functools import partial
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import skimage.measure

LOG = logging.getLogger(__name__)

class EMnistDataset:
    """
    Loads the MNIST data saved in .npy or .npz files.

    If the 'labels' argument is left as None then the class assumes that the file
    in 'data' is .npz and creates attributes, with the same name as specified
    during the file creation, containing the respective numpy arrays.

    If the 'labels' argument is set to a string path then the class assumes that
    the files were saved as .npy and it will create two attributes: 'imgs' which
    contains the contents of the 'data' file and 'labels' with the contents of
    the 'labels' file.

    If you chose to save the arrays differently then you might have to modify
    this class or write your own loader.
    """

    SPACE_CODE = 62

    def __init__(self, data : str = "emnist_train.npz", labels : str = None):

        if not os.path.exists(data):
            raise ValueError("Requested mnist data file not found!")
        if (labels is not None) and (not os.path.exists(labels)):
            raise ValueError("Requested mnist label file not found!")

        if labels is None:
            dataset = np.load(data)
            for key, value in dataset.items():
                setattr(self, key, value)
        else:
            self.imgs = np.load(data)
            self.labels = np.load(labels)


class Phrases:

    phrases = [
        "move forward", "go forward", "go straight", "roll forward", "stumble on", "shamble on", "walk forward",
        "turn left", "wheel left", "rotate left", "spin left",
        "turn right", "wheel right", "rotate right", "spin right",
        "turn back", "wheel back", "rotate back", "spin back", "turn around", "wheel around",
        "move left", "walk left", "shamble left", "go leftish", "stumble left", "skulk left",
        "move right", "walk right", "roll right", "go rightish", "skulk right",
        "move back", "walk back", "shamble back", "go backward", "stumble back", "skulk back",
    ]

    commands = ["F", "L", "R", "B", "ML", "MR", "MB"]

    phraseToCommand = ["F"] * 7 + ["L"] * 4 + ["R"] * 4 + ["B"] * 6 + ["ML"] * 6 + ["MR"] * 5 + ["MB"] * 6


@dataclass(frozen=True)
class Bound:
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def get_bounds(self) -> tuple[int, int, int, int]:
        return self.x_min, self.x_max, self.y_min, self.y_max

    def num_cols(self) -> int:
        # coordinates (x, y): array indexing, NOT axis-coordnicates
        return self.y_max - self.y_min

    def num_rows(self) -> int:
        # coordinates (x, y): array indexing, NOT axis-coordnicates
        return self.x_max - self.x_min

    def get_bounded_img(self, img: np.ndarray) -> np.ndarray:
        return img[self.x_min: self.x_max, self.y_min: self.y_max]


class ImagePreprocessor():
    '''
    Preprocessing entails:
        - finding bounding boxes around objects in the image
        - cropping the bounding boxes, and padding them adequately for training
    '''

    @staticmethod
    def _get_obj_bounds_in_img(img: np.ndarray) -> list[Bound]:
        '''returns the unordered bounds of the objects in the image'''
        img_height, img_width = img.shape
        x_coords, y_coords    = np.mgrid[:img_width, :img_height]

        img_copy               = img.copy()
        img_copy[img_copy > 0] = 1
        labelled_chunks_img    = skimage.measure.label(img_copy)

        # we first just skip dots in i or chuncks that are too small
        def bounding_box(label: int) -> Bound:
            labelled_ids = labelled_chunks_img == label

            labelled_xs = x_coords[ labelled_ids ]
            labelled_ys = y_coords[ labelled_ids ]

            return Bound(
                labelled_xs.min(), labelled_xs.max(),
                labelled_ys.min(), labelled_ys.max()
            )

        def big_enough(bound: Bound) -> bool:
            xa, xb, ya, yb = bound.get_bounds()
            return xb - xa > 10 and yb - ya > 3

        min_label, max_label = 1, labelled_chunks_img.max()
        object_bounds = []
        for label in range(min_label, max_label + 1):
            bounds = bounding_box(label)
            if big_enough(bounds):
                object_bounds.append(bounds)

        return object_bounds


    @staticmethod
    def _with_blank_space_bound(line: list[Bound]) -> list[Bound]:
        '''
        Add a space bound, which will be expanded in img resizing
        in _get_resized_img_from_bound_with_img(...)
        '''

        # if the width between two characters is bigger than delta,
        # we add a space
        space_delta = 50

        for i, (bound_i, bound_j) in enumerate( zip(line, line[1:]) ):
            if bound_j.y_min - bound_i.y_max > space_delta:
                line.insert(i + 1, Bound(0, 0, 0, 0))
                break

        return line


    @staticmethod
    def _reorder_bounds_to_phrases_and_lines(object_bounds: list[Bound]) -> list[list[Bound]]:
        '''return ordered bounds separated by lines and ordered within a line'''
        first_col_bound: Optional[Bound] = None
        line_epsilon = 14

        lines = []
        line = []

        def flush_line() -> None:
            nonlocal line, lines
            if line:
                LOG.debug(f'adding {len(line)} words to lines')
                ordered_line = sorted(line, key=lambda bound: bound.y_min)
                ordered_line_with_blank_space = ImagePreprocessor._with_blank_space_bound(ordered_line)
                lines.append(ordered_line_with_blank_space)
            line = []

        for object_bound in object_bounds:
            if first_col_bound is None or abs(first_col_bound.x_min - object_bound.x_min) > line_epsilon:
                first_col_bound = object_bound
                flush_line()

            line.append(object_bound)

        flush_line()

        return lines

    @staticmethod
    def _get_resized_img_from_bound_with_img(img: np.ndarray, bound: Bound) -> np.ndarray:
        '''
        given a bound, obtain the img chunk enclosed in the bound,
        and resize it to a 28x28 image
        '''
        obj_chuck_in_img = bound.get_bounded_img(img)

        curr_cols, curr_rows = bound.num_cols(), bound.num_rows()
        x_axis_pad_left  = (28 - curr_cols) // 2
        x_axis_pad_right = 28 - (curr_cols + x_axis_pad_left)

        y_axis_pad_top  = (28 - curr_rows) // 2
        y_axis_pad_botm = 28 - (curr_rows + y_axis_pad_top)

        LOG.debug(f'chunk_shape: {obj_chuck_in_img.shape}; cc: {curr_cols}, cr: {curr_rows}, ({x_axis_pad_left} - {x_axis_pad_right}), ({y_axis_pad_top} | {y_axis_pad_botm})')
        vert_padded_img = np.r_[
            np.zeros((y_axis_pad_top, curr_cols)),
            obj_chuck_in_img,
            np.zeros((y_axis_pad_botm, curr_cols))
        ]
        total_padded_img = np.c_[
            np.zeros((28, x_axis_pad_left)), vert_padded_img, np.zeros((28, x_axis_pad_right))
        ]

        return total_padded_img

    @staticmethod
    def get_phrases_and_lines(img: np.ndarray) -> list[list[np.ndarray]]:
        '''return ordered character images from ordered phrases in img lines'''
        unordered_img_char_bounds  = ImagePreprocessor._get_obj_bounds_in_img(img)
        ordered_img_char_bounds    = ImagePreprocessor._reorder_bounds_to_phrases_and_lines(unordered_img_char_bounds)
        get_resized_img_from_bound = partial(ImagePreprocessor._get_resized_img_from_bound_with_img, img)

        return [
            [
                get_resized_img_from_bound(bound)
                for bound in line
            ]
            for line in ordered_img_char_bounds
        ]
