# NOTE: You should not edit this source file because it is not part of your solution.
# You will hand in your solution without this file and if you submit it, it will be ignored.
# - So, any changes that break the original evaluation will look as mistakes in your solution.

import argparse
import logging
import os
from pathlib import Path
import re
from typing import Sequence, Tuple, List, Dict

import numpy as np

import page_reader

# * Set up logging
class CustomFormatter(logging.Formatter):
    grey     = "\x1b[38;20m"
    yellow   = "\x1b[33;20m"
    red      = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset    = "\x1b[0m"
    log_fmt  = "\n%(levelname)s: (%(filename)s:%(lineno)d): %(message)s"

    FORMATS = {
        logging.DEBUG:      grey + log_fmt + reset,
        logging.INFO:       grey + log_fmt + reset,
        logging.WARNING:    yellow + log_fmt + reset,
        logging.ERROR:      red + log_fmt + reset,
        logging.CRITICAL:   bold_red + log_fmt + reset
    }

    def format(self, record):
        log_fmt   = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setFormatter(CustomFormatter())

LOG = logging.getLogger()
LOG.addHandler(CONSOLE_HANDLER)
LOG.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="single", type=str, help="Type of evaluation; one of 'single' for evaluation of one task or 'full' for evaluation of an entire set.")
parser.add_argument("--set", default="python_train", type=str, help="Name of the evaluated set - it is a directory containing maze runs.")
parser.add_argument("--name", default="001.npz", type=str, help="Name of the evaluated maze run in the selected set for type 'single'.")
parser.add_argument("--training", default=False, action="store_true", help="Calls training routine before evaluation.")
parser.add_argument("--verbose", default=2, type=int, help="Level of verbosity: '0' - prints only the final result, '1' - prints every maze run result, '2' - prints every page result.")
parser.add_argument("--note", default=None, type=str, help="String note passed to the solution in its constructor - used as the name of the model file.")

class MazeRunLoader:
    """Loads the maze run data - pages and labels."""

    def read(self, filename : str) -> Dict[str, object]:
        with np.load(filename, allow_pickle=True) as mazeRunHandle:
            mazeRun = dict(mazeRunHandle)
        return mazeRun

    def completePageText(self, sourceText : Sequence[Sequence[str]], numbers : Sequence[str]) -> List[List[str]]:
        """Adds numbers to the page text for per-character evaluation."""
        text = []
        for pageText, number in zip(sourceText, numbers):
            text.append(list(pageText))
            text[-1].append(number)
        return text

class Evaluator:
    """Provides functions for maze run evaluation."""

    def __init__(self, args : argparse.Namespace) -> None:
        self.args = args
        self.loader = MazeRunLoader()
        self.solution = page_reader.PageReader(self.args.note)
        self.pointIdx = { "text" : 0, "phrase" : 1, "number" : 2, "path" : 3 }

    def _compareLists(self, firstList : Sequence[object], secondList : Sequence[object]) -> int:
        points = 0
        for i in range(min(len(firstList), len(secondList))):
            if firstList[i] == secondList[i]:
                points += 1
        return points

    def _compareCharacters(self, detectedText : Sequence[str], trueText : Sequence[str]) -> Tuple[int, int]:
        maxPoints = max(np.sum([len(t) for t in detectedText]), np.sum([len(t) for t in trueText]))
        points = 0
        for i in range(min(len(detectedText), len(trueText))):
            points += self._compareLists(detectedText[i], trueText[i])
        return points, maxPoints

    def _compareWords(self, detectedWords : Sequence[str], trueWords : Sequence[str]) -> Tuple[int, int]:
        maxPoints = max(len(detectedWords), len(trueWords))
        points = self._compareLists(detectedWords, trueWords)
        return points, maxPoints

    def _printResuls(self, name : str, points : np.ndarray, maxPoints : np.ndarray) -> None:
        print("{} per-character accuracy: {:.2f}%".format(name, points[self.pointIdx["text"]] / maxPoints[self.pointIdx["text"]] * 100))
        print("{} per-phrase accuracy:    {:.2f}%".format(name, points[self.pointIdx["phrase"]] / maxPoints[self.pointIdx["phrase"]] * 100))
        print("{} page number accuracy:   {:.2f}%".format(name, points[self.pointIdx["number"]] / maxPoints[self.pointIdx["number"]] * 100))
        print("{} path (final) accuracy:  {:.2f}%".format(name, points[self.pointIdx["path"]] / maxPoints[self.pointIdx["path"]] * 100))

    def _evaluateRun(self, mazeRun : Dict[str, np.ndarray], name : str):
        """Evaluates one loaded maze run."""
        trueText = self.loader.completePageText(mazeRun["text"], mazeRun["numbers"])
        detectedText, detectedPhrases, detectedPath = self.solution.solve(mazeRun["pages"])
        # Score counting.
        points, maxPoints = np.zeros((4), dtype=int), np.zeros((4), dtype=int)
        maxPoints[self.pointIdx["number"]] = len(trueText)
        # Process each page separately.
        for pDetectedText, pDetectedPhrases, pTrueText, pTruePhrases in zip(detectedText, detectedPhrases, trueText, mazeRun["text"]):
            (textPoints, maxTextPoints), (phrasePoints, maxPhrasePoints) = self._compareCharacters(pDetectedText, pTrueText), self._compareWords(pDetectedPhrases, pTruePhrases)
            points[self.pointIdx["text"]], maxPoints[self.pointIdx["text"]] = points[self.pointIdx["text"]] + textPoints, maxPoints[self.pointIdx["text"]] + maxTextPoints
            points[self.pointIdx["phrase"]], maxPoints[self.pointIdx["phrase"]] = points[self.pointIdx["phrase"]] + phrasePoints, maxPoints[self.pointIdx["phrase"]] + maxPhrasePoints
            points[self.pointIdx["number"]] += pDetectedText[-1] == pTrueText[-1]
            if self.args.verbose > 1:
                print("Page '{}' statistics".format(pTrueText[-1]))
                print("Per-character page accuracy: {:.2f}%".format(points[self.pointIdx["text"]] / maxPoints[self.pointIdx["text"]] * 100))
                print("Per-phrase page accuracy:    {:.2f}%".format(points[self.pointIdx["phrase"]] / maxPoints[self.pointIdx["phrase"]] * 100))
                print("Detected number/True number: {}/{}".format(pDetectedText[-1], pTrueText[-1]))
        # Compute points for the final path.
        points[self.pointIdx["path"]], maxPoints[self.pointIdx["path"]] = self._compareWords(detectedPath, mazeRun["path"])
        if self.args.verbose > 0:
            print("Maze run '{}' statistics".format(name))
            self._printResuls("Maze run", points, maxPoints)
        return points, maxPoints

    def _evaluateMazeRuns(self, name : str, mazeRunFiles : Sequence[str]) -> None:
        """Evaluates a set of maze run tasks."""
        mazeRuns = []
        for f in mazeRunFiles:
            mazeRuns.append(self.loader.read(f))

        totalPoints, totalMaxPoints = np.zeros((4), dtype=int), np.zeros((4), dtype=int)
        for i, mazeRun in enumerate(mazeRuns):
            points, maxPoints = self._evaluateRun(mazeRun, Path(mazeRunFiles[i]).stem)
            totalPoints, totalMaxPoints = totalPoints + points, totalMaxPoints + maxPoints
        print("Overall evaluation of {}.".format(name))
        self._printResuls("Total", totalPoints, totalMaxPoints)

    def _evaluateSingle(self) -> None:
        """Evaluates one maze run with loading of file selected through arguments."""
        fileName = self.args.name if self.args.name.endswith(".npz") else self.args.name + ".npz"
        evaluatedFile = os.path.join("page_data", self.args.set, fileName)
        self._evaluateMazeRuns("the maze run '{}'".format(fileName), [evaluatedFile])

    def _evaluateFull(self):
        """Evaluates all maze runs found in the directory specified through arguments."""
        evaluatedFolder = os.path.join("page_data", self.args.set)
        regex = re.compile('.*\.npz$')
        mazeRunFiles = []
        dirList = os.listdir(evaluatedFolder)
        for f in dirList:
            if regex.match(f):
                mazeRunFiles.append(f)
        mazeRunFiles = [os.path.join(evaluatedFolder, f) for f in mazeRunFiles]
        self._evaluateMazeRuns("the set '{}'".format(self.args.set), mazeRunFiles)

    def evaluate(self):
        """Evaluates the tasks requested through arguments."""
        self.solution.fit(self.args.training)
        evaluation = {
            "single" : self._evaluateSingle,
            "full" : self._evaluateFull,
        }
        if self.args.type not in evaluation:
            raise ValueError("Unrecognised type of evaluation: '{}', please, use one of: 'single'/'full'.".format(self.args.type))
        evaluation[args.type]()

def main(args : argparse.Namespace):
    evaluator = Evaluator(args)
    evaluator.evaluate()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
