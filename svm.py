# %%
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASET_DIR: str = "./ESC-50/"
AUDIO_DIR: str = DATASET_DIR + "audio/"
META_DIR: str = DATASET_DIR + "meta/"
META_FILE: str = META_DIR + "esc50.csv"
FEATURES_FILE: str = "./features_means.csv"

# %%
# Load dataset metadata and features
files: pd.DataFrame = pd.read_csv(META_FILE)
features: pd.DataFrame = pd.read_csv(FEATURES_FILE)
features.rename(columns={"file_name": "filename"}, inplace=True)
# Merge features and metadata
df: pd.DataFrame = pd.merge(files, features, on="filename", how="inner")


# %%
# Setup folds
folds = [
    {
        "train": df.loc[df["fold"] != i, :],
        "val": df.loc[df["fold"] == i, :],
    }
    for i in df["fold"].unique()
]

fold_sizes = [
    {
        "train": len(df[df["fold"] != i]),
        "val": len(df[df["fold"] == i]),
    }
    for i in df["fold"].unique()
]

# %%
# Train and evaluate SVM
accuracies = []
for fold in folds:
    # Split into train and validation
    train_df = fold["train"]
    val_df = fold["val"]

    # Extract features
    train_features = train_df.drop(
        ["filename", "fold", "target", "category", "esc10", "src_file", "take"], axis=1
    )
    val_features = val_df.drop(
        ["filename", "fold", "target", "category", "esc10", "src_file", "take"], axis=1
    )

    # Extract labels
    train_labels = train_df["target"]
    val_labels = val_df["target"]

    # Train SVM
    clf = make_pipeline(
        StandardScaler(),
        PCA(n_components="mle"),
        GridSearchCV(
            estimator=SVC(
                C=1.0,
                kernel="rbf",
                degree=3,
                gamma="scale",
                coef0=0.0,
                probability=True,
                cache_size=1000,
                decision_function_shape="ovr",
                break_ties=True,
            ),
            param_grid={
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
                "gamma": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
    )
    clf.fit(train_features, train_labels)

    # Evaluate SVM
    val_predictions = clf.predict(val_features)
    accuracy = sklearn.metrics.accuracy_score(val_labels, val_predictions)
    accuracies.append((accuracy, clf["gridsearchcv"].best_params_))

print(accuracies)
# %%

# %%
