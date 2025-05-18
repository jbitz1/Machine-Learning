import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the dataset
data = pd.read_csv('income_data.csv')

data.columns = data.columns.str.strip()
print(data.info())

# Drop unnecessary column
data = data.drop(columns=['fnlwgt', 'education'])

# Verify the columns have been dropped
print(data.info())
# Identify categorical and numerical features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# List of categorical columns to encode
categorical_columns = ["workclass", "marital-status", "occupation",
                       "relationship", "race", "sex", "native-country"]

rows_with_question_mark = data.apply(lambda row: row.astype(str).str.contains(r'\?').any(), axis=1)

data = data[~rows_with_question_mark]

labelEncoder = preprocessing.LabelEncoder()
for column in categorical_columns:
    data[column] = labelEncoder.fit_transform(data[column])

X = data.drop(columns=['income'])  # Features
y = data['income']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate Decision Tree model using ID3 algorithm (criterion='entropy')
random_forest = RandomForestClassifier()

# Train the classifier (fitting the model)
random_forest.fit(X_train, y_train)

# Make predictions
y_pred = random_forest.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

