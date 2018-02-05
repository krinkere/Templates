from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import tree

# Load and split the data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Construct pipeline
pipe = Pipeline([('scl', StandardScaler()),
                 ('pca', PCA(n_components=2)),
                 ('clf', tree.DecisionTreeClassifier(random_state=42))])

# Fit the pipeline
pipe.fit(X_train, y_train)

# # Pipeline estimator params; estimator is stored as step 3 ([2]), second item ([1])
# print('Default available hyperparameters:', pipe.steps[2][1].get_params())
# # Pipeline test accuracy
# print('Default accuracy: %.3f' % pipe.score(X_test, y_test))

param_range = [1, 2, 3, 4, 5]

# Set grid search params
grid_params = [{'clf__criterion': ['gini', 'entropy'],
                'clf__min_samples_leaf': param_range,
                'clf__max_depth': param_range,
                'clf__min_samples_split': param_range[1:],
                'clf__presort': [True, False]}]

# Construct grid search
gs = GridSearchCV(estimator=pipe,
                  param_grid=grid_params,
                  scoring='accuracy',
                  cv=10)

# Fit using grid search
gs.fit(X_train, y_train)

# # Best params
# print('Best hyperparameters:', gs.best_params_)
# # Best accuracy
# print('Best accuracy: %.3f' % gs.best_score_)

print('Default accuracy: %.3f' % pipe.score(X_test, y_test))
print('criterion:', pipe.steps[2][1].get_params().get("criterion"))
print('max_depth:', pipe.steps[2][1].get_params().get("max_depth"))
print('min_samples_leaf:', pipe.steps[2][1].get_params().get("min_samples_leaf"))
print('min_samples_split:', pipe.steps[2][1].get_params().get("min_samples_split"))
print('presort:', pipe.steps[2][1].get_params().get("presort"))

print('\nBest accuracy: %.3f' % gs.best_score_)
print('criterion:', gs.best_params_.get('clf__criterion'))
print('max_depth:', gs.best_params_.get('clf__max_depth'))
print('min_samples_leaf:', gs.best_params_.get('clf__min_samples_leaf'))
print('min_samples_split:', gs.best_params_.get('clf__min_samples_split'))
print('presort:', gs.best_params_.get('clf__presort'))
