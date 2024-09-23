import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


train_file_path = '/Users/ivermalme/Desktop/train.csv'
test_file_path = '/Users/ivermalme/Desktop/test.csv'


train_data = pd.read_csv(train_file_path)
test_data_with_id = pd.read_csv(test_file_path)


X_train = train_data.drop(['Id', 'Target'], axis=1)
y_train = train_data['Target']
X_test = test_data_with_id.drop('Id', axis=1)


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


gb_classifier_weighted = GradientBoostingClassifier(random_state=42)
gb_classifier_weighted.fit(X_train_split, y_train_split, sample_weight=y_train_split.map({0: 1, 1: 3}))


y_val_pred_weighted = gb_classifier_weighted.predict(X_val_split)


accuracy_weighted = accuracy_score(y_val_split, y_val_pred_weighted)
classification_rep_weighted = classification_report(y_val_split, y_val_pred_weighted)


print(f'Accuracy: {accuracy_weighted}')
print('Classification Report:')
print(classification_rep_weighted)


test_predictions = gb_classifier_weighted.predict(X_test)


test_results = pd.DataFrame({
    'Id': test_data_with_id['Id'],
    'Target': test_predictions
})


test_results.to_csv('predictions.csv', index=False)


print(test_results)