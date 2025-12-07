import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


# here you can change the dataset path 
# data = pd.read_csv(r"alldata.csv")
data = pd.read_csv(r"thermal_statistics_alldata.csv")

data['class'] = data['class'].str.strip()
data = data[data['class'].isin(['Sick', 'Healthy'])]

print("Dataset Information:") 
print(f"Total samples: {len(data)}") 
print("Class distribution:") 
print(data['class'].value_counts()) 
print()

label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

# here you can change the features used for training
# X = data[['Average Temperature', 'Min Temperature', 'Max Temperature', 'Most Frequent Temperature']]

X = data[['Average Temperature','Min Temperature','Max Temperature','Standard Deviation'
  ,'25th Percentile','Median (50th Percentile)','75th Percentile','IQR','Skewness','Kurtosis','Mode']]
y = data['class']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Data Split Information:")
print(f"Training samples:   {len(X_train)} ({len(X_train)/len(data)*100:.1f}%)") 
print(f"Validation samples: {len(X_val)}  ({len(X_val)/len(data)*100:.1f}%)") 
print(f"Testing samples:    {len(X_test)} ({len(X_test)/len(data)*100:.1f}%)\n")

print("Training set class distribution:")
for label, count in pd.Series(y_train).value_counts().items():
    print(f"  {label_encoder.inverse_transform([label])[0]}: {count}")

print("\nTesting set class distribution:")
for label, count in pd.Series(y_test).value_counts().items():
    print(f"  {label_encoder.inverse_transform([label])[0]}: {count}")
print()

models = {
    "Neural Network": MLPClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(random_state=42),
    "K-NN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
}

for name, model in models.items():
    print("\n-----------------------------------------")
    print(f"Training model: {name}")
    print("-----------------------------------------")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} - Accuracy: {accuracy:.4f}")
    train_acc = model.score(X_train, y_train)
    val_acc   = model.score(X_val, y_val)
    test_acc  = model.score(X_test, y_test)
    print(f"\nAccuracy Results for: {name}")
    print(f"Training Accuracy:   {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")

    # Classification Report (per class)
    print("\nClassification Report:")
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4
    )
    print(report)
    cm = confusion_matrix(y_test, y_pred)

    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()

        print(f"\nModel: {name} (Accuracy: {accuracy:.4f})")
        print("Detailed Model Metrics:")
        print(f"True Positives (TP): {TP}")
        print(f"True Negatives (TN): {TN}")
        print(f"False Positives (FP): {FP}")
        print(f"False Negatives (FN): {FN}")

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.4f}") 
        print(f"Recall (Sensitivity): {recall:.4f}") 
        print(f"Specificity: {specificity:.4f}") 
        print(f"F1-Score: {f1:.4f}")

    else:
        print(f"Model: {name}")
        print("Confusion Matrix shape:", cm.shape)
        print("Confusion Matrix:")
        print(cm)

    # Plot confusion matrix  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues, values_format='d') 
    plt.title(f"Confusion Matrix - {name}\nAccuracy: {accuracy:.4f}") 
    plt.tight_layout()
    plt.show()
