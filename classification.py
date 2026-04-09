from preprocessing import classification_pp # returns attributes and target (X, y) for regression dataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

def classification():
    """
    needs to find class probabilites, not predictions, so we can evaluate AUC and choose a threshold for precision/recall
    Use evaluation function to evaluate performance of different models
    """
    X_train, X_test, y_train, y_test = classification_pp()

    # 10-fold cross-validation
    X = X_train
    y = y_train

    # Majority Class ==================================================================
    print("1). Majority Class Classifier")
    model = DummyClassifier(strategy='most_frequent')
    y_probs = cross_val_predict(model, X, y, cv=10, method='predict_proba')[:, 1]
    evaluation(y_probs, y)

    # Decision Tree ==================================================================
    print("\n2). Decision Tree")
    dt_models = [
        DecisionTreeClassifier(max_depth=None),
        DecisionTreeClassifier(criterion='entropy'),
        DecisionTreeClassifier(max_depth=5),
        DecisionTreeClassifier(min_samples_split=10),
    ]

    for model in dt_models:
        print(f"\nModel: {model}")
        y_probs = cross_val_predict(model, X, y, cv=10, method='predict_proba')[:, 1]
        evaluation(y_probs, y)

    # Random Forests ==================================================================
    print("\nRandom Forest")

    rf_models = [
        RandomForestClassifier(),
        RandomForestClassifier(n_estimators=200),
        RandomForestClassifier(n_estimators=50),
        RandomForestClassifier(max_depth=10),
    ]

    for model in rf_models:
        print(f"\nModel: {model}")
        y_probs = cross_val_predict(model, X, y, cv=10, method='predict_proba')[:, 1]
        evaluation(y_probs, y)

    return

def evaluation(y_probs, y_true, *, threshold=0.5, verbose=True):
    """
    Evaluates classification performance using various metrics.
    Returns a tuple of (accuracy, precision, recall, AUC, confusion matrix).
    Takes in predicted class PROBABILITIES (y_probs) and true labels (y_true), along with a threshold for converting probabilities to binary predictions.
    """
    y_pred = (y_probs >= threshold).astype(int)  # Convert probabilities to binary predictions

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0) # I've added the zero_division thingy to prevent warning doo-dad
    recall = recall_score(y_true, y_pred, zero_division=0) # Here too
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.show()

    return (accuracy, precision, recall, auc, cm)


def main():
    classification()


if __name__ == "__main__":
    main()