# Set UP
car_sales = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales-extended.csv")
car_sales.head()

x = car_sales.drop("Price", axis=1)
y = car_sales["Price"]

# split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

# Transform categorical data to numerical data
# OneHot --------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make","Colour","Doors"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_features)],
                                  remainder="passthrough")
transformed_x = transformer.fit_transform(x)
transformed_x

# Get Dummies ------------------------------------------------------------------------------------------------------------------------------
dummies = pd.get_dummies(car_sales[["Make", "Colour","Doors"]])
dummies



# Then you can train the model
np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(transformed_x,
                                                    y,
                                                    test_size=0.2)
model.fit(x_train, y_train)
