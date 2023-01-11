
import argparse
import logging
import lzma
import os
import pickle
import multiprocessing
from typing import Tuple, List

import numpy as np
import scipy.ndimage
import sklearn.base
import sklearn.ensemble
import sklearn.neural_network

import utils

LOG = logging.getLogger(__name__)

# NOTE: Arguments in this source file are only for your convenience. They will not be used during
# evaluation but you can use them for easier experimentation with your solution.
parser = argparse.ArgumentParser()
parser.add_argument("--example", default=None, type=int, help="Example argument.")

class PageReader:

    def __init__(self, note : str) -> None:
        # TODO: Place any initialisation of your method in here. If your method doesn't require
        # initialisation then you can keep this method empty but you have to remove 'raise NotImplementedError()'.
        # Argument 'note' is passed to this method by the evaluator. It is a single string argument that you
        # can use for something, e.g., name of the trained/loaded model in the fit method.
        self.note = note or 'my_model'
        self.model_filename = f'{self.note}.model'

        # build the by class mapping
        self.label_to_chr_code = {}
        self.chr_code_to_label = {}
        class_mapping_file = './emnist-byclass-mapping.txt'
        if not os.path.exists(class_mapping_file):
            raise FileNotFoundError('byclass mapping file not found!')
        else:
            with open(class_mapping_file, 'r') as f:
                for line in f:
                    a, b = map(int, line.split())
                    self.label_to_chr_code[a] = b
                    self.chr_code_to_label[b] = a

        assert len(self.label_to_chr_code) == 63, \
            LOG.error(f'if {class_mapping_file} 62 mappings only, it is missing the mapping for a space character')


    def _load_model(self) -> None:
        with lzma.open(self.model_filename, 'rb') as model_file:
            self.digit_clf, self.letter_clf = pickle.load(model_file)


    def _augment(self, img: np.ndarray) -> np.ndarray:
        '''augment the image to produce more training samples'''
        img = img.reshape(28, 28)
        img = scipy.ndimage.zoom(img.reshape(28, 28), (np.random.uniform(0.86, 1.2), np.random.uniform(0.86, 1.2)))
        img = np.pad(img, ((2, 2), (2, 2)))

        rnds = [np.random.randint(size - 28 + 1) for size in img.shape]

        img = img[rnds[0]:rnds[0] + 28, rnds[1]:rnds[1] + 28]
        img = scipy.ndimage.rotate(img, np.random.uniform(-15, 15), reshape=False)
        img = np.clip(img, 0, 1)

        return img.reshape(-1)


    def _fit_clf(self, x_train: np.ndarray, y_train: np.ndarray, augment=True, num_models=3) -> sklearn.base.ClassifierMixin:
        '''fit an MLP ensemble for x_train and y_train'''
        iterations = 12
        model = sklearn.ensemble.VotingClassifier([
                (
                    f"MLP{i}",
                    sklearn.neural_network.MLPClassifier(
                        tol=0, verbose=1, alpha=0, hidden_layer_sizes=(250),
                        max_iter=iterations
                    )
                )
                for i in range(num_models)
            ], voting="soft"
        )
        model.fit(x_train, y_train)

        def harmonize(y_train: np.ndarray, classes: np.ndarray) -> np.ndarray:
            '''
            the labels passed to partial_fit without this modification threw an error
            saying the mlp.classes_ had different classes than y_train, which was weird.
            This harmonize routine assumes that y_train and uniq_classes contain all possible labels.
            '''
            uniq_classes = np.unique(classes)
            uniq_y_train = np.unique(y_train)
            for y_val, cls in zip(uniq_y_train, uniq_classes):
                y_train[y_train == y_val] = cls

            return y_train

        if augment:
            pool = multiprocessing.Pool(16)
            for mlp in model.estimators_:
                aug_y_train = harmonize(y_train, mlp.classes_)
                for epoch in range(iterations - 1):
                    LOG.info(f"Augmenting data for epoch {epoch}...")
                    augmented_data = pool.map(self._augment, x_train)
                    LOG.info("Done")
                    mlp.partial_fit(augmented_data, aug_y_train)

        return model


    def _add_space_data(self, letter_x_train: np.ndarray, letter_y_train: np.ndarray) -> None:
        '''Add adequate number of empty images to train letter model for spaces'''
        num_letter_samples = 1000
        letter_x_train = np.r_[letter_x_train, np.zeros((num_letter_samples, 28 * 28))]
        letter_y_train = np.r_[
            letter_y_train, np.full(num_letter_samples, utils.EMnistDataset.SPACE_CODE)
        ].astype(int)

        return letter_x_train, letter_y_train


    def _retrain_model(self) -> None:
        '''train the digit and letter models and save them'''

        # train digits model
        digit_data = utils.EMnistDataset('./emnist_data/emnist_digits_selection_train.npz')

        # preprocess digit data
        dig_x_train, dig_y_train = digit_data.imgs, digit_data.labels
        dig_x_train = dig_x_train.reshape(-1, 28 * 28)

        self.digit_clf = self._fit_clf(dig_x_train, dig_y_train)

        # train letter model
        letter_data = utils.EMnistDataset('./emnist_data/emnist_small_letters_train.npz')

        # preprocess letter data
        letter_x_train, letter_y_train = letter_data.imgs, letter_data.labels
        letter_x_train = letter_x_train.reshape(-1, 28 * 28)
        letter_x_train, letter_y_train = self._add_space_data(letter_x_train, letter_y_train)

        self.letter_clf = self._fit_clf(letter_x_train, letter_y_train)

        # compress MLPs
        for voting_ensemble in (self.digit_clf, self.letter_clf):
            for mlp in voting_ensemble.estimators_:
                mlp._optimizer = None
                for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
                for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        with lzma.open(self.model_filename, 'wb') as model_file:
            pickle.dump((self.digit_clf, self.letter_clf), model_file)


    def fit(self, training : bool) -> None:
        # TODO: Place your training, model saving and model loading code in here.
        # This method will be called once before any evaluation takes place so you should set up all objects
        # required for the task solving.
        # For example, running evaluation of a complete set of tasks will be done by creating an instance
        # of this class, calling this 'fit' method and then calling 'solve' for every task.
        # >>> pr = PageReader()
        # >>> pr.fit(args.training)
        # >>> pr.solve(task1)
        # >>> pr.solve(task2) etc.
        #
        # This method should be able to train your solution on demand. That means, if the argument 'training'
        # is True then you should train your classification models and use the newly trained ones. If the argument
        # is False then you should load models from saved files.
        [

            self._load_model,
            self._retrain_model

        ][training]()


    def solve(self, pages : np.ndarray) -> Tuple[List[List[str]], List[List[str]], List[str]]:
        # TODO: This method should solve a single page reading task.
        # It gets a stack of page images on its input. You have to process and classify all pages
        # and return the text you extracted from each page, phrases which you matched on each page
        # (both in the input order of the pages) and the final list of movement instructions in the correct order
        # according to the page numbers.

        def list_to_num(dig_list: np.ndarray) -> int:
            return int(np.sum(np.geomspace(10 ** (len(dig_list) - 1), 1, len(dig_list)) * dig_list))

        page_number_preds = []
        page_texts = []
        for page in pages:
            # predict the text / characters, and phrases
            lines = utils.ImagePreprocessor.get_phrases_and_lines(page)

            # the last line is the line number
            char_lines = lines[:-1]
            page_text  = []
            for line in char_lines:
                # decode the predicted label the legible character
                line = [img.reshape(28 * 28) for img in line] # preprocess line images
                char_line_pred = self.letter_clf.predict(line)
                decoded_pred   = [
                    chr(self.label_to_chr_code[label]) for label in char_line_pred
                ]
                # page_texs: list[list[str]], ie, list of list of lines, so we join the line words.
                # no need for space, as model predicts the space.
                page_text.append(''.join(decoded_pred))

            page_number_imgs = [
                img.reshape(28 * 28) for img in lines[-1]
            ] # preprocess number images
            assert len(page_number_imgs),  LOG.error(f'expecting 4 page number digits, got {len(page_number_imgs)}')
            page_number_pred = self.digit_clf.predict(page_number_imgs)

            page_number = list_to_num(page_number_pred)
            page_number_preds.append(page_number)

            page_text.append(f'{page_number:04}')
            page_texts.append(page_text)

        # find the closest instructions for each page, and rearrange them by the page numbers
        pool = multiprocessing.Pool(processes=2)
        cmd_proc    = pool.apply_async(func=self._find_commands_and_rearrange_them_using, args=(page_texts, page_number_preds))
        phrase_proc = pool.apply_async(func=self._find_phrases_using, args=(page_texts,))

        # we need to return all text, matched phrases, and commands
        # all text includes the page numbers
        return page_texts, phrase_proc.get(), cmd_proc.get()


    def _find_phrases_using(self, page_texts: list[list[str]]) -> list[list[str]]:
        # skip the page numbers
        pages = [
            page[:-1] for page in page_texts
        ]

        phrases_by_page = []
        for page in pages:
            page_phrases = []
            for phrase_pred in page:
                def hamming(phrase: str):
                    dist = 0
                    for c1, c2 in zip(phrase_pred, phrase):
                        dist += (c1 != c2)
                    return dist
                closest_phrase = min(utils.Phrases.phrases, key=hamming)
                page_phrases.append(closest_phrase)

            phrases_by_page.append(page_phrases)

        return phrases_by_page


    def _find_commands_and_rearrange_them_using(self,
        page_texts: list[list[str]], page_nums: list[int]
        ) -> list[str]:
        '''return the commands rearranged by page numbers'''

        # flatten page_text from pages of list of lines to pages of text without the page-number
        pages = [
            ' '.join(page[:-1]) for page in page_texts
        ]

        # sort pages by page numbers
        sorted_pages = [
            page for _, page in sorted(zip(page_nums, pages))
        ]
        # the split removes all whitespace, leaving us solely with the phrase words
        sorted_commands = ' '.join(sorted_pages).split()

        # find commands by hamming distances
        commands = []
        for word1, word2 in zip(sorted_commands[0::2], sorted_commands[1::2]):
            # without the page number, the command phrases come in pairs
            def hamming(word_index: int):
                dist = 0
                for c1, c2 in zip(f'{word1} {word2}', utils.Phrases.phrases[word_index]):
                    dist += (c1 != c2)
                return dist

            closest_phrase_index = min(range(len(utils.Phrases.phrases)), key=hamming)
            closest_command = utils.Phrases.phraseToCommand[closest_phrase_index]
            commands.append(closest_command)

        LOG.debug(commands)
        return commands


def main(args : argparse.Namespace) -> None:
    # NOTE: You can run any test or visualisation that you want here or anywhere else.
    # However, you must not change the signature of 'PageReader.__init__', 'PageReader.fit' or 'PageReader.solve'.
    #
    # Your solution will be evaluated using 'evaluator.py' as it was given to you. This means
    # that you should not change anything in 'evaluator.py'. Also, you should make sure that
    # your solution can be evaluated with on-demand training.
    #
    # Evaluation of your solution through the commandline can look like this:
    # >>> python ./evaluator.py --type=single --set=python_train --name=005 --verbose=2 --note=something
    # >>> python ./evaluator.py --type=full --set=python_train --verbose=1 --training
    # >>> python ./evaluator.py --type=full --set=python_validation --verbose=1
    pass


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
