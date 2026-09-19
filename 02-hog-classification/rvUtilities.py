from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import ml_utilities
from joblib import Memory

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy
import sklearn.metrics

def getTestLabels():
    test_path = 'DBs/CaniGatti_ML18/Unlabeled_BinaryTestSet.txt'

    test_df = pd.read_table(test_path, delimiter="/", names=["path1","path2","filename"])
    preds = list()
    i = 0
    while i < len(test_df):
        pred = 0 if test_df["filename"][i][0:3]=='cat' else 1
        preds.append(pred)
        i = i + 1
    test_labels = list()
    for file in test_df["filename"]:
          label = 0 if file[0:3]=='cat' else 1
          test_labels.append(label)
    return test_labels

def printAccuracy(y_true, y_pred):
  acc = sklearn.metrics.accuracy_score(y_true, y_pred)
  print(acc)


def useGridSearchToCalculateBestHyperparams(model_params, train_feature_x, train_y, test_size, n_split):
    result = None
    for data_test_size in test_size: #0.2,0.25,0.3
        for split in n_split: #, 2, 3, 4, 5, 6, 7, 8, 9, 10
            cross_val = StratifiedShuffleSplit(n_splits=split, test_size=data_test_size, random_state=42)

            for model_name, mp in model_params.items():
                grid = GridSearchCV(estimator=mp['model'],
                                    param_grid=mp['params'],
                                    cv=cross_val,
                                    verbose=2,
                                    return_train_score=False)


                grid.fit(train_feature_x, train_y)

                print("Iteration score: ")
                print(grid.best_score_)
                if result == None or result.best_score_ < grid.best_score_:
                    print("Best score replaced: ")
                    result = grid
                    print(result.best_score_)
    
    return result



def useVotingClassifierWithGridSearchCV(X, y):
  clf1 = LogisticRegression(random_state=1)
  clf2 = RandomForestClassifier(random_state=1)
  clf3 = GaussianNB()
  eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft'
  )

  params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}

  grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
  grid = grid.fit(X, y)
  print(grid.score(X, y))


# Bagging methods work best with strong and complex models
# see https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator
def useBaggingClassifier(clf, train_X, train_y, test):
  clf = BaggingClassifier(base_estimator=clf,
                          max_samples=0.5, max_features=0.5,
                          n_estimators=10, random_state=0).fit(train_X, train_y)

  predictions = clf.predict(test)
  print("Accuratezza: ", clf.score(train_X, train_y))
  return predictions

def useDecisionTreeClassifier(X, y):
  clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
        random_state=0)
  scores = cross_val_score(clf, X, y, cv=5)
  print(scores.mean())

def useRandomForestClassifier(X, y):
  clf = RandomForestClassifier(n_estimators=10, max_depth=None,
  min_samples_split=2, random_state=0)
  scores = cross_val_score(clf, X, y, cv=5)
  print(scores.mean())


def useExtraTreesClassifier(X, y):
  clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                             min_samples_split=2, random_state=0)
  scores = cross_val_score(clf, X, y, cv=5)
  print(scores.mean())


def useAdaBoostClassifier(X, y):
  clf = AdaBoostClassifier(n_estimators=100)
  scores = cross_val_score(clf, X, y, cv=5)
  print(scores.mean())



def useMajorityVotingClassifier(X, y):
  clf1 = LogisticRegression(random_state=1, max_iter=500)
  clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
  clf3 = GaussianNB()
  clf4 = BaggingClassifier(SVC(), max_samples=0.5, max_features=0.5,
                          n_estimators=10, random_state=0)

  eclf = VotingClassifier(
     estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('bc', clf4)],
     voting='hard')

  for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Bagging Classifier', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



# https://scikit-learn.org/stable/modules/ensemble.html#xgboost
def useSoftVotingClassifier(X, y):
  clf1 = DecisionTreeClassifier(max_depth=4)
  clf2 = KNeighborsClassifier(n_neighbors=7)
  clf3 = SVC(kernel='rbf', probability=True)
  eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
                          voting='soft', weights=[2, 1, 2])

  print("=== Soft voting classifier (probabilities) ===")
  print(clf1.fit(X, y).score(X, y))
  print(clf2.fit(X, y).score(X, y))
  print(clf3.fit(X, y).score(X, y))
  print(eclf.fit(X, y))
  print("Ensambled classifier score: ", eclf.score(X, y))




def getTestData(image_side):
  memory = Memory('Experiments', verbose=0)
  test_filelist = 'Unlabeled_BinaryTestSet.txt'
  db_path = 'DBs/CaniGatti_ML18'

  test_raw_x = ml_utilities.load_unlabeled_dataset(test_filelist, db_path, cache=memory)

  test_raw_x = ml_utilities.resize_images(test_raw_x, image_side, image_side, cache=memory)

  test_raw_x = ml_utilities.extract_hog(test_raw_x, 
                                        convert_to_gray=True, orientations=9,
                                        pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                                        cache=memory)
  return test_raw_x


def getTrainData(image_side):
  db_path = 'DBs/CaniGatti_ML18'
  train_filelist = 'BinaryTrainingSet.txt'  
  memory = Memory('Experiments', verbose=0) 

  train_raw_x, train_y = ml_utilities.load_labeled_dataset(train_filelist, db_path, cache=memory)

  train_raw_x = ml_utilities.resize_images(train_raw_x, image_side, image_side, cache=memory)

  train_feature_x = ml_utilities.extract_hog(train_raw_x, 
                                           convert_to_gray=True, orientations=9,
                                           pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                                           cache=memory)
  return train_feature_x, train_y


def getTrainData12500(image_side):
  db_path = 'DBs/dogs-vs-cats/'
  train_filelist = 'train2.txt'  
  memory = Memory('Experiments', verbose=0) 

  train_raw_x, train_y = ml_utilities.load_labeled_dataset(train_filelist, db_path, cache=memory)

  train_raw_x = ml_utilities.resize_images(train_raw_x, image_side, image_side, cache=memory)

  train_feature_x = ml_utilities.extract_hog(train_raw_x, 
                                           convert_to_gray=True, orientations=9,
                                           pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                                           cache=memory)
  return train_feature_x, train_y


def getTestData(image_side):
  db_path = 'DBs/CaniGatti_ML18'
  test_filelist = 'Unlabeled_BinaryTestSet.txt' 
  memory = Memory('Experiments', verbose=0) 

  test_raw_x = ml_utilities.load_unlabeled_dataset(test_filelist, db_path, cache=memory)

  test_raw_x = ml_utilities.resize_images(test_raw_x, image_side, image_side, cache=memory)

  test_feature_x = ml_utilities.extract_hog(test_raw_x, 
                                           convert_to_gray=True, orientations=9,
                                           pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                                           cache=memory)
  return test_feature_x


def trainAndGetOptimalClassifier(X, y):
  bclf = SVC()
  clf0 = LogisticRegression(solver='lbfgs', max_iter=1000)
  clf1 = SVC(kernel='rbf', probability=True)
  clf2 = RandomForestClassifier(random_state=1)
  clf3 = AdaBoostClassifier(n_estimators=100)
  eclf = VotingClassifier(estimators=[('lr', clf0), ('svc', clf1), ('rf', clf2), ('adaboost', clf3)],
                          voting='soft', weights=[1, 1, 1, 1])

  print("=== Soft voting classifier (probabilities) ===")
  print("BaggingClassifier: ", clf0.fit(X, y).score(X, y))
  print("SVC: ", clf1.fit(X, y).score(X, y))
  print("RandomForestClassifier: ", clf2.fit(X, y).score(X, y))
  print("AdaBoostClassifier: ", clf3.fit(X, y).score(X, y))
  print("Ensambled classifier score: ", eclf.fit(X, y).score(X, y))

  params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}

  grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, verbose=2)
  grid = grid.fit(X, y)
  print("GridSearchCV:", grid.score(X, y))
  return grid
