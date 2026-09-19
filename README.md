# Machine learning labs

[![Build](https://github.com/davidcohenDC/machine-learning-labs/actions/workflows/build.yml/badge.svg)](https://github.com/davidcohenDC/machine-learning-labs/actions/workflows/build.yml)
[![Release](https://img.shields.io/github/v/release/davidcohenDC/machine-learning-labs)](https://github.com/davidcohenDC/machine-learning-labs/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Five Jupyter labs from the Machine Learning course at the University of
Bologna, by [David Cohen](https://github.com/davidcohenDC) and
[Giulia Nardicchia](https://github.com/GiuliaNardicchia). Every lab ends with a
prediction file scored on a hidden test set, and the scores add up into a
course-wide leaderboard: **Team Savignano, second overall**, among the five
teams that earned exam bonus points.

Razvan Vasile worked through labs 2 and 3 with us without being on the
registered team; his commits are in the history.

## The labs

Each folder is one lab: the course notebook with our cells in it, sometimes a
second notebook where the real work happened, the helper modules, the data and
the predictions we submitted.

- **[01-svm-knn](01-svm-knn)**: handwritten digits (PenDigits, 16 features).
  Grid searches over SVM, k-NN and logistic regression on several stratified
  splits, 513 configurations logged to a table (`results/`). The best is an
  RBF SVM at 0.93 cross-validated accuracy;
- **[02-hog-classification](02-hog-classification)**: cats versus dogs from
  HOG features. We added the 25 000 Kaggle *dogs-vs-cats* images and rotated
  copies (45°, 60°, 90°, 180°) to the course set: 28 700 patterns, 75 minutes
  of HOG extraction, then SVM, Random Forest and AdaBoost. `hog-opencv-variant`
  is the same lab with OpenCV descriptors and a voting classifier;
- **[03-solar-regression](03-solar-regression)**: power of a solar park one
  hour ahead, scored on RMSE. `pre_processing.py` is ours: calendar features,
  cyclic encodings of hour and day, parts of the day, a hand-made list of bad
  rows to drop. `final-model` runs a Random Forest grid and averages the
  predictions of the winners;
- **[04-audio-mlp](04-audio-mlp)**: spoken digits from MFCC, chroma, Mel and
  spectral-contrast features. A deep MLP in Keras trained across 15 stratified
  folds with early stopping;
- **[05-cnn-finetuning](05-cnn-finetuning)**: twelve pet breeds, MobileNet V1
  backbone fixed by the rules. Augmentation, then five networks from a 5-fold
  split and a majority vote: 74.5% on the test set.

Labs 2 and 5 read images that lived on the lab machines and are not here, so
they can be read but not re-run. The other three run in the image below.

## Run it

You need Docker. Then:

```sh
git clone https://github.com/davidcohenDC/machine-learning-labs.git
cd machine-learning-labs
docker compose up
```

and open http://localhost:8888: JupyterLab with the Python stack of late 2022
(TensorFlow 2.10, scikit-learn 1.1, pinned in `requirements.txt`). Compose
pulls the image from GitHub Packages and builds it only if the pull fails.

To run a notebook top to bottom without opening it:

```sh
docker compose run --rm lab scripts/run.sh 01-svm-knn/svm-knn.ipynb
```

The run is written next to the notebook as `<name>.run.ipynb`, so the
committed file stays as it is. Lab 1 takes a minute, `03-solar-regression/
regression.ipynb` a few; `final-model.ipynb` sweeps a large Random Forest
grid and is a matter of hours, `04-audio-mlp/competition.ipynb` trains a wide
MLP fifteen times and is the same on a CPU. The CI runs lab 1 and lab 3 on
every push.

Prefer not to clone? Each
[release](https://github.com/davidcohenDC/machine-learning-labs/releases/latest)
ships a zip with everything.

## How it is organised

```
01-svm-knn/             svm-knn.ipynb, grid-search-experiments.ipynb, classification.py, results/, DBs/PenDigits
02-hog-classification/  hog-classification.ipynb, hog-opencv-variant.ipynb, rvUtilities.py, DBs/dogs-vs-cats
03-solar-regression/    regression.ipynb, final-model.ipynb, pre_processing.py, DBs/SolarPark
04-audio-mlp/           neural-networks.ipynb, competition.ipynb, utilities.py, DBs/AudioDigits_ML21_22
05-cnn-finetuning/      cnn-finetuning.ipynb, competition.ipynb, DBs/CaniGatti_ML18_Es8_2020 (file lists only)
scripts/run.sh          executes one notebook with nbconvert
Dockerfile, compose.yaml, requirements.txt
```

`ml_utilities.py` and `ml_visualization.py` in each folder are the course's
helper modules; `predictions.txt` is what we uploaded to the leaderboard. The
datasets are the ones handed out in class.

## License

[MIT](LICENSE) © 2022 David Cohen, Giulia Nardicchia
