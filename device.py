import xmlrpc.client
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Connect to the RPC server (dispatcher)
with xmlrpc.client.ServerProxy("http://192.168.1.234:9000/") as proxy:
    print("Fetching MNIST data from Dispatcher...")
    data, target = proxy.get_mnist()

    # Convert data back to numpy arrays
    data = np.array(data)
    target = np.array(target)

    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, shuffle=False)

    # Train an SVM model
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(X_train, y_train)

    predicted = classifier.predict(X_test)

    print(f"Classification report:\n{metrics.classification_report(y_test, predicted)}")
