# remove warnings
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

pd.options.display.max_rows = 100

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def load_data():
    # reading train and test data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    return combined, targets


def feature_engineer(data):
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

    data = process_age_missing_valuse(data)

    data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)

    return data

    combined.drop('Name', axis=1, inplace=True)


def process_age_missing_valuse(data):
    # imputation age by linear model PClass, Sibsp, Parch, Fare
    age_train = data[pd.notnull(data['Age'])]
    x = age_train[['Pclass', 'SibSp', 'Parch']]
    y = age_train['Age']

    # from sklearn import linear_model
    # regr = linear_model.LinearRegression()
    # regr.fit(x, y)
    # data.Age = data.apply(lambda r: regr.predict(r[['Pclass', 'SibSp', 'Parch']])
    #                       if np.isnan(r['Age']) else r['Age'], axis=1)

    etr = ExtraTreesRegressor(n_estimators=200)
    etr.fit(x, np.ravel(y))

    age_preds = etr.predict(data[['Pclass', 'SibSp', 'Parch']][data['Age'].isnull()])
    data['Age'][data['Age'].isnull()] = age_preds

    return data


def build_model(data, targets):
    size = targets.shape[0]
    train = data.ix[0:size - 1]
    test = data.ix[size + 1:]

    # Feature selection
    clf = ExtraTreesClassifier(n_estimators=200)
    clf = clf.fit(train, targets)

    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_
    features.sort(['importance'], ascending=False)

    # Model
    forest = RandomForestClassifier(max_features='sqrt')
    parameter_grid = {
        'max_depth': [4, 5, 6, 7, 8],
        'n_estimators': [200, 210, 240, 250],
        'criterion': ['gini', 'entropy']
    }

    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               n_jobs=4)

    grid_search.fit(train, targets)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))


if __name__ == '__main__':
    data, targets = load_data()

    data = feature_engineer(data)

    build_model(data, targets)
