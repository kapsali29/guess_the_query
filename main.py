from process import read_and_clean_training_data, clean_test_data, _train_model, _predict

x_train, y_train = read_and_clean_training_data("training.txt")
clf = _train_model(x_train, y_train)

xtest, ytest = clean_test_data("x_test.txt", "y_test.txt")
accuracy = _predict(xtest, ytest, clf)
print(accuracy)
