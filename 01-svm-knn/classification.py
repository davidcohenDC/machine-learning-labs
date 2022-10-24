from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import ml_utilities


class BestParams:
    _bestparams = None
    _t_size = 0
    _split = 0

    def update(self, bestparams, t_size, split):
        self._bestparams = bestparams
        self._t_size = t_size
        self._split = split

    def isbetter(self, score):
        if self._bestparams is None: return True
        # print(score.cv_results_['mean_test_score'], self._bestparams['cv_results_'])
        return score.cv_results_['mean_test_score'] > self._bestparams.cv_results_['mean_test_score']

    def printVals(self):
        print("_bestparams: ", self._bestparams.best_params_)
        print("_t_size: ", self._t_size)
        print("_split: ", self._split)


def normalizeData(dataset_patterns, test_x):
    scaler = MinMaxScaler()
    # alldata = np.concatenate((dataset_patterns, test_x))
    # transformed_data = scaler.fit_transform(alldata)

    transformed_data_train = scaler.fit_transform(dataset_patterns)
    transformed_data_test = scaler.fit_transform(test_x)

    # dataset_patterns = transformed_data[:442]
    # test_x = transformed_data[442:]
    return transformed_data_train, transformed_data_test


def readData():
    feature_count = 16
    dataset_path = 'DBs/PenDigits/pendigits_tr.txt'  # Impostare il percorso corretto

    dataset_patterns, dataset_labels = ml_utilities.load_labeled_dataset_from_txt(dataset_path, feature_count)
    test_path = 'DBs/PenDigits/pendigits_te.txt'
    test_x = ml_utilities.load_unlabeled_dataset_from_txt(test_path, feature_count)
    return dataset_patterns, dataset_labels, test_x


def getOttimalClassifier(bestparams, dataset_patterns, dataset_labels):
    for t_size in [0.1, 0.13, 0.16, 0.21, 0.23, 0.25, 0.28, 0.30, 0.32, 0.33, 0.35, 0.37, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for split in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print(f"TEST = {t_size}%, SPLIT = {split}")
            gamma = [2 ** i for i in range(-20, 10)]
            gamma = np.append(gamma, 0.00009111627561154887)

            C = [2 ** i for i in range(-7, 20)]
            C = np.append(C, 1)

            # degree = [i for i in range(1,30)]
            param_grid = [{'kernel': ['rbf'], 'C': C, 'gamma': gamma}]
            # param_grid = [{'kernel': ['poly'], 'C': C, 'gamma': gamma, 'degree': [5,6,7,8,9,10]}]

            cross_val = StratifiedShuffleSplit(n_splits=split, test_size=t_size, random_state=42)
            grid_search_cv = GridSearchCV(SVC(), param_grid, cv=cross_val, verbose=2)
            grid_search_cv.fit(dataset_patterns, dataset_labels)

            if bestparams.isbetter(grid_search_cv):
                bestparams.update(grid_search_cv, t_size, split)

    return grid_search_cv


def main():
    print('ciao')
    bestparams = BestParams()
    # Caricamento del dataset
    feature_count = 16
    dataset_path = 'DBs/PenDigits/pendigits_tr.txt'  # Impostare il percorso corretto

    dataset_patterns, dataset_labels = ml_utilities.load_labeled_dataset_from_txt(dataset_path, feature_count)
    data_patterns, data_labels, test_x = readData()
    dataset_patterns, test_x = normalizeData(dataset_patterns, test_x)
    print()

    grid_search_cv = getOttimalClassifier(bestparams, dataset_patterns, dataset_labels)
    print(grid_search_cv.cv_results_['mean_test_score'])
    bestparams.printVals()

    clf = SVC(**grid_search_cv.best_params_, random_state=42)
    # Addestramento del classificatore
    clf.fit(data_patterns, data_labels)


# the specifications of any data necessary for the execution of the tests
if __name__ == "__main__":
    main()
