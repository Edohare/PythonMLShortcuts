from sklearn.model_selection import RandomizedSearchCV

# Create a dictionary of the hyperparameters we would like to adjust
grid = {"n_estimators": [10, 100, 200, 500, 1000, 1200],
        "max_depth": [None, 5, 10, 20, 30],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4]}

np.random.seed(42)

# split into x and y
# change to your data
x = heart_disease_shuffled.drop('target', axis = 1)
y = heart_disease_shuffled['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# instantiate RandomForestClassifier
clf = RandomForestClassifier(n_jobs=1)

# Set up RandomizedSearchCV
rs_clf = RandomizedSearchCV(estimator = clf, 
                            param_distributions = grid,
                            n_iter = 10, # Number of models to try
                            cv = 5, # cross fold validation
                            verbose = 2)

# Fit the RandomizedSearchCV version of clf
rs_clf.fit(x_train, y_train);

# Make predicitons with the best hyper_parameters
rs_y_preds = rs_clf.predict(x_test)

# Evaluate the predicitions
rs_metrics = evaluate_preds(y_test, rs_y_preds)
