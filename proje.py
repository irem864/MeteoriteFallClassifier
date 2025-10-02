import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("meteorite-landings.csv")
df.dropna(inplace=True)


data = pd.get_dummies(df, columns=['recclass'])
label_encoder = LabelEncoder()
data['fall_encoded'] = label_encoder.fit_transform(df['fall'])


X = data.drop(['name', 'nametype', 'fall', 'fall_encoded', 'GeoLocation'], axis=1)
y = data['fall_encoded']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return preds, probs

models = {
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier()
}

results = {}
for name, model in models.items():
    preds, probs = train_model(model, X_train, y_train, X_test)
    results[name] = {
        "preds": preds,
        "probs": probs,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average='weighted'),
        "recall": recall_score(y_test, preds, average='weighted'),
        "f1": f1_score(y_test, preds, average='weighted'),
        "roc_auc": roc_auc_score(y_test, probs) if probs is not None else None,
        "conf_matrix": confusion_matrix(y_test, preds)
    }


performance_data = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]["accuracy"] for m in results],
    "Precision": [results[m]["precision"] for m in results],
    "Recall": [results[m]["recall"] for m in results],
    "F1 Score": [results[m]["f1"] for m in results],
    "ROC AUC Score": [results[m]["roc_auc"] for m in results]
})

print(performance_data)


metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']
for metric in metrics:
    plt.figure(figsize=(5,5))
    ax = sns.barplot(x='Model', y=metric, data=performance_data)
    plt.title(metric)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width()/2, p.get_height()), ha='center', va='bottom')
    plt.show()


plt.figure(figsize=(6,6))
for name in models:
    if results[name]["probs"] is not None:
        fpr, tpr, _ = roc_curve(y_test, results[name]["probs"])
        plt.plot(fpr, tpr, label=name)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


for name in models:
    plt.figure(figsize=(6,4))
    sns.heatmap(results[name]["conf_matrix"], annot=True, fmt=".0f", cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
