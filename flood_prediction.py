import pandas as pd 

df =pd.read_csv("flood_data.csv", sep="\t")
print(df.head())

X = df[[
    "Rainfall","Elevation","Slope"
]]

y = df["FloodRisk"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators= 100,
    random_state= 42
)

model.fit(X_train, y_train)

print("Model Successfully Trained!!")

y_pred = model.predict(X_test)

print(y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)


results = X_test.copy()

results["Actual"] = y_test
results["Predicted"] = y_pred

print(results)

importance = model.feature_importances_

for feature, score in zip(X.columns, importance):
    print(feature, ":", round(score, 3))