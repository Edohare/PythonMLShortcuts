from sklearn.model_selection import GridSearchCV, train_test_split

# Grid search tests every combination of your grid, instead of a random number of combinations

grid_2 = {'n_estimators': [100, 200, 500],
          'max_depth': [None],
          'max_features': ['auto', 'sqrt'],
          'min_samples_split': [6],
          'min_samples_leaf': [1, 2]}

np.random.seed(42)

# split into x and y
# change to your data
x = heart_disease_shuffled.drop('target', axis = 1)
y = heart_disease_shuffled['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# instatiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Set up RandomizedSearchCV
gs_clf = GridSearchCV(estimator = clf, 
                      param_grid = grid_2,
                      # Do not need n_iter because grid search tries every combination
                      cv = 5, # cross fold validation
                      verbose = 2)

# Fit the GridSearchCV version of clf
gs_clf.fit(x_train, y_train);

# get best params
gs_clf.best_params_

# Make predicitons with the best hyper_parameters
gs_y_preds = gs_clf.predict(x_test)

# Evaluate the predicitions
gs_metrics = evaluate_preds(y_test, gs_y_preds)
