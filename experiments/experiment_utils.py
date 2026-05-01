from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    # treino + temporário
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # validação + teste
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(y_true, y_pred, verbose=True):
    if verbose:
        print(classification_report(y_true, y_pred))

    f1 = f1_score(y_true, y_pred, average="macro")
    return f1