import xmlrpc.client
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

# Connect to the RPC server (dispatcher)
with xmlrpc.client.ServerProxy("http://192.168.1.234:9090/RPC2") as proxy:
    print("Fetching MNIST data from Dispatcher...")
    data, target = proxy.get_mnist()

    # Convert data back to numpy arrays
    data = np.array(data)[:int(0.4*len(data))]  # Only take 40% of the dataset
    target = np.array(target)[:int(0.4*len(target))]

    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, shuffle=False)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a Linear SVM model
    classifier = LinearSVC(max_iter=10000)
    classifier.fit(X_train, y_train)

    # Dump the trained model to a file
    joblib.dump(classifier, 'trained_model.pkl')

    # Send the model back to the dispatcher (This is a simplified approach; in practice, you might need more intricate handling)
    with open('trained_model.pkl', 'rb') as model_file:
        proxy.upload_model(xmlrpc.client.Binary(model_file.read()))

    # Make predictions on the test set
    predicted = classifier.predict(X_test)

    # Print evaluation results
    print(f"Accuracy: {accuracy_score(y_test, predicted)}")
    print(f"F1 Score: {f1_score(y_test, predicted, average='macro')}")
    print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_test, predicted)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predicted)}")
