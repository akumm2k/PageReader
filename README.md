# Maze Run Page Reader

The model is trained on pages containing handwritten instructions to solve a Maze.
The letters and digits of the pages come from the [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## Training Notes

- Model: Voting ensemble of MLPs, partially fitted over randomly augmented images.
- The letter classifier is trained on the alphabet and the space character, so, I added the class mapping for the space character to `emnist-byclass-mapping.txt` in the end. Therefore, my `emnist-byclass-mapping.txt` is necessary - don't remove it.
- the filename of the model to be trained goes in the `--note` option of the evaluator.

## Training Time

Trained the model on a computer with `Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz`, 8-cores and 16 GB of RAM

Time taken:
```
> time python3 evaluator.py --note='mlps2' --training 1>/dev/null
2337.04s user
352.19s system
688% cpu 6:30.69 total
```

## Evaluation Usgae

The name of the model to be trained / loaded goes into the `note` flag of the evaluator.

Example Run:
```
> python3 ./evaluator.py --type single --set=python_validation --name=001 --verbose=2 --note='mlps'
Page '1983' statistics
Per-character page accuracy: 95.38%
Per-phrase page accuracy:    100.00%
Detected number/True number: 1983/1983
Page '0305' statistics
Per-character page accuracy: 91.40%
Per-phrase page accuracy:    88.24%
Detected number/True number: 0305/0305
Page '1154' statistics
Per-character page accuracy: 89.63%
Per-phrase page accuracy:    92.00%
Detected number/True number: 1154/1154
Page '0245' statistics
Per-character page accuracy: 90.15%
Per-phrase page accuracy:    93.33%
Detected number/True number: 0245/0245
Page '1147' statistics
Per-character page accuracy: 90.77%
Per-phrase page accuracy:    94.29%
Detected number/True number: 1147/1147
Page '1825' statistics
Per-character page accuracy: 90.62%
Per-phrase page accuracy:    95.00%
Detected number/True number: 1825/1825
Page '1672' statistics
Per-character page accuracy: 88.52%
Per-phrase page accuracy:    94.00%
Detected number/True number: 1672/1672
Page '1564' statistics
Per-character page accuracy: 88.11%
Per-phrase page accuracy:    94.64%
Detected number/True number: 1564/1564
Maze run '001' statistics
Maze run per-character accuracy: 88.11%
Maze run per-phrase accuracy:    94.64%
Maze run page number accuracy:   100.00%
Maze run path (final) accuracy:  96.43%
Overall evaluation of the maze run '001.npz'.
Total per-character accuracy: 88.11%
Total per-phrase accuracy:    94.64%
Total page number accuracy:   100.00%
Total path (final) accuracy:  96.43%
```
