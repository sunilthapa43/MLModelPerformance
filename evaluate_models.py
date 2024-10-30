import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, accuracy_score

from tensorflow.keras.utils import to_categorical



def evaluate_model(model, X_train, X_test, y_train, y_test):
    if isinstance(model, (tf.keras.models.Sequential, tf.keras.Model)):
        # Determine if it’s binary or multiclass classification
        num_classes = len(np.unique(y_train))

        # Prepare labels
        if num_classes > 2:
            y_train_cat = to_categorical(y_train)  # For multiclass classification
            model.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=1)
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs > 0.5).astype(int)
        else:
            # For binary classification
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

    else:
        # For non-deep learning models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Compute metrics
    fpr, recall, f1, precision, accuracy = compute_metrics(y_test, y_pred)
    return fpr, recall, f1, precision, accuracy


# Compute metrics
def compute_metrics(y_true, y_pred):
    if len(set(y_true)) <= 2:
        # Binary classification
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
    else:
        # Multiclass classification: Use macro-average for FPR calculation
        conf_matrix = confusion_matrix(y_true, y_pred)
        fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  # False Positives for each class
        tn = conf_matrix.sum() - (fp + conf_matrix.sum(axis=1) - np.diag(conf_matrix))
        fpr = np.mean(fp / (fp + tn))  # Mean FPR for all classes

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    return fpr, recall, f1, precision, accuracy
