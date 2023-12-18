from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def ROC_curve(model, X_test, y_test, model_name):
    RocCurveDisplay.from_predictions(
        y_test.to_numpy(),
        model.predict_proba(X_test)[:, 1],
        name=f"ROC Curve - {model_name}",
        color="darkorange",
    )
    plt.plot([0,1],[0,1], "k--")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc = 'lower right')
    plt.show()


def confusion_mat(model, y_hat, y_test, model_name):
    cm = confusion_matrix(y_test, y_hat, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


def correlation_plots():
    pass