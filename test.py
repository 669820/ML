import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the train and test data
train_file_path = '/Users/ivermalme/Desktop/train.csv'  # Update path
test_file_path = '/Users/ivermalme/Desktop/test.csv'    # Update path

# Load the datasets
train_data = pd.read_csv(train_file_path)
test_data_with_id = pd.read_csv(test_file_path)

# Prepare train and test data by dropping the 'Id' column for features
X_train = train_data.drop(['Id', 'Target'], axis=1)
y_train = train_data['Target']
X_test = test_data_with_id.drop('Id', axis=1)

# Split the training data into training and validation sets (80% train, 20% validation)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier with class weighting to handle imbalance
gb_classifier_weighted = GradientBoostingClassifier(random_state=42)
gb_classifier_weighted.fit(X_train_split, y_train_split, sample_weight=y_train_split.map({0: 1, 1: 3}))

# Predict on the validation set with the weighted model
y_val_pred_weighted = gb_classifier_weighted.predict(X_val_split)

# Evaluate the model performance after applying class weighting
accuracy_weighted = accuracy_score(y_val_split, y_val_pred_weighted)
classification_rep_weighted = classification_report(y_val_split, y_val_pred_weighted)

# Print the results
print(f'Accuracy: {accuracy_weighted}')
print('Classification Report:')
print(classification_rep_weighted)

# Make predictions on the test set
test_predictions = gb_classifier_weighted.predict(X_test)

# Prepare the results in a DataFrame for output
test_results = pd.DataFrame({
    'Id': test_data_with_id['Id'],  # Use test_data_with_id to ensure 'Id' is available
    'Target': test_predictions
})

# Save the results to a CSV file
test_results.to_csv('predictions.csv', index=False)

# Display the test predictions
print(test_results)
