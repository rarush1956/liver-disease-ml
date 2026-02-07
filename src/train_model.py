from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from preprocessing import load_and_preprocess

def train():
    X, y = load_and_preprocess("data/Indian_Liver_Patient_Dataset.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    model.fit(X_res, y_res)
    return model, X_test, y_test

if __name__ == "__main__":
    train()