from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

#Shuffle the data, since we are creating our train, test and validation manually
# This takes in a sample of the data, frac=1 is 100% of the data, and shuffles it.
heart_disease_shuffled = heart_disease.sample(frac=1)

# Split into x & y
x = heart_disease_shuffled.drop('target', axis=1)
y = heart_disease_shuffled['target']

# split the data into train, val & test sets

train_split = round(0.7 * len(heart_disease_shuffled)) # 70% of the data
valid_split = round(train_split + 0.15 * len(heart_disease_shuffled)) # 15% of data
x_train, y_train = x[:train_split], y[:train_split]
x_valid, y_valid = x[train_split:valid_split], y[train_split:valid_split]
x_test, y_test = x[valid_split:], y[valid_split:]

clf = RandomForestClassifier()

clf.fit(x_train, y_train)

# make predictions
y_preds = clf.predict(x_valid)

# Evaluate the classifier on validations et
baseline_metrics = evaluate_preds(y_valid, y_preds)
baseline_metrics
