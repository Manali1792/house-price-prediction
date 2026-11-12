from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def create_preprocessor(scale_features=True):

    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_features:
        steps.append(
            ("scaler", StandardScaler())
        )

    preprocessor = Pipeline(steps)

    return preprocessor
