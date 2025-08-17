from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

drive.mount('/content/drive/')

!ls /content/drive/MyDrive/

df = pd.read_csv('/content/drive/MyDrive/diabetest.csv')

print(df.shape)
df.head()
df.info()

df.isnull().sum()

df.describe()

df.groupby('Outcome').mean()

df.hist(bins=20, figsize=(12,10))
plt.tight_layout()
plt.show()

import seaborn as sns

features_to_plot = ['Glucose', 'BMI']
plt.figure(figsize=(12, 12))
sns.boxplot(data=df[features_to_plot])
plt.title("Boxplot of Selected Features", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome', palette='coolwarm')
plt.title('Scatter plot of Glucose vs BMI')

from sklearn.linear_model import LinearRegression

# انتخاب فقط یک ویژگی
X = df[['BMI']]
y = df['Glucose']

# آموزش مدل
model = LinearRegression()
model.fit(X, y)

# پیش‌بینی خط رگرسیون
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# رسم نمودار نقطه‌ای + خط رگرسیون
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.6, label='Actual Data')
plt.plot(x_range, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('Linear Regression: BMI vs Glucose')
plt.legend()
plt.grid(True)
plt.show()

#پیش‌بینی مقادیر با مدل آموزش دیده
y_pred = model.predict(X)  # استفاده از همان مدل رگرسیون خطی قبلی

#ایجاد یک دیتافریم برای مقایسه
comparison = pd.DataFrame({
    'مقدار واقعی': y,  # مقادیر واقعی Glucose
    'مقدار پیش‌بینی شده': y_pred  # مقادیر پیش‌بینی شده توسط مدل
})

#نمایش 10 نمونه تصادفی
print(comparison.sample(10))  #نمایش 10 رکورد تصادفی برای مقایسه

#رسم نمودار مقایسه ای
plt.figure(figsize=(10, 5))
plt.scatter(y, y_pred, alpha=0.5)  # نقاط آبی: مقایسه واقعی vs پیش‌بینی
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # خط قرمز: حالت ایده آل
plt.xlabel('Actual values of Glucose')
plt.ylabel('Predicted values')
plt.title('Comparing actual and predicted values')
plt.grid(True)
plt.show()

#برای محاسبه خطا
comparison['خطا'] = comparison['مقدار واقعی'] - comparison['مقدار پیش‌بینی شده']
print(comparison.head())

# انتخاب ویژگی‌ها برای مدل SVM
X_svm = df[['BMI', 'Glucose']]
y_svm = df['Outcome']

# آموزش مدل SVM با کرنل خطی
from sklearn.svm import SVC
model_svm = SVC(kernel='linear')
model_svm.fit(X_svm, y_svm)

# پیش‌بینی
y_pred_svm = model_svm.predict(X_svm)

from sklearn.inspection import DecisionBoundaryDisplay

plt.figure(figsize=(10, 6))

# رسم مرز تصمیم‌گیری
DecisionBoundaryDisplay.from_estimator(
    model_svm,
    X_svm,
    cmap='coolwarm',
    alpha=0.4,
    response_method="predict",
)

# رسم داده‌های واقعی
plt.scatter(
    X_svm['BMI'],
    X_svm['Glucose'],
    c=y_svm,
    cmap='coolwarm',
    edgecolors='k',
    s=50,
    alpha=0.8
)

plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('SVM decision boundary (linear kernel)')
plt.colorbar(label='Outcome (0: non-diabetic, 1: diabetic)')
plt.grid(True)
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# محاسبه معیارها
accuracy = accuracy_score(y_svm, y_pred_svm)
precision = precision_score(y_svm, y_pred_svm)
recall = recall_score(y_svm, y_pred_svm)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# رسم گرافیکی ماتریس سردرگمی
cm_linear = confusion_matrix(y_svm, y_pred_svm)
disp_linear = ConfusionMatrixDisplay(confusion_matrix=cm_linear, display_labels=["No Diabetes", "Diabetes"])
disp_linear.plot(cmap='Blues')
plt.title('Confusion Matrix — SVM (Linear with Combined Feature)')
plt.grid(False)
plt.show()

# آموزش مدل با کرنل RBF
from sklearn.svm import SVC
model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_svm, y_svm)

# پیش‌بینی
y_pred_rbf = model_rbf.predict(X_svm)

# ایجاد مش برای ناحیه تصمیم
x_min, x_max = X_svm.iloc[:, 0].min() - 1, X_svm.iloc[:, 0].max() + 1
y_min, y_max = X_svm.iloc[:, 1].min() - 1, X_svm.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# رسم ناحیه تصمیم و داده‌های واقعی
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_svm.iloc[:, 0], X_svm.iloc[:, 1], c=y_svm, cmap='coolwarm', edgecolors='k')
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('SVM Decision Boundary — RBF Kernel')
plt.grid(True)
plt.show()

# محاسبه معیارها
accuracy = accuracy_score(y_svm, y_pred_rbf)
precision = precision_score(y_svm, y_pred_rbf)
recall = recall_score(y_svm, y_pred_rbf)

print(f'RBF SVM — Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# رسم ماتریس درهم‌ریختگی
cm_rbf = confusion_matrix(y_svm, y_pred_rbf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rbf, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap='Purples')
plt.title('Confusion Matrix — SVM (RBF)')
plt.grid(False)
plt.show()

from sklearn.preprocessing import StandardScaler

# انتخاب ویژگی‌ها
features = ['BMI', 'Glucose', 'Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
X_raw = df[features]

# ایجاد و آموزش اسکیلر
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

from sklearn.ensemble import IsolationForest

# ایجاد مدل Isolation Forest
iso_model = IsolationForest(contamination=0.05, random_state=42)

# آموزش مدل و پیش‌بینی ناهنجاری‌ها
df['Anomaly_ISO'] = (iso_model.fit_predict(X_scaled) == -1).astype(int)

# جداسازی ناهنجاری‌ها
anomalies = df[df['Anomaly_ISO'] == 1]

from sklearn.neighbors import NearestNeighbors

# تنظیم تعداد همسایه‌ها
k = 5

# ایجاد مدل KNN
knn = NearestNeighbors(n_neighbors=k)

# آموزش مدل روی داده‌های استاندارد شده
knn.fit(X_scaled)

# محاسبه فاصله هر نقطه از k همسایه نزدیکش
distances, _ = knn.kneighbors(X_scaled)

# محاسبه میانگین فاصله برای هر نقطه
mean_distances = distances.mean(axis=1)

# تعیین آستانه ناهنجاری (صدک 95ام)
threshold = np.percentile(mean_distances, 95)

# علامت‌گذاری ناهنجاری‌ها
df['Anomaly_KNN'] = (mean_distances > threshold).astype(int)

from scipy.stats import zscore

# محاسبه z-score برای داده‌های استاندارد شده
z_scores = np.abs(zscore(X_scaled))

# علامت‌گذاری نقاط با z-score بالاتر از 3 در هر یک از ویژگی‌ها
df['Anomaly_Z'] = (z_scores > 3).any(axis=1).astype(int)

# محاسبه چارک‌ها
Q1 = df[features].quantile(0.25)  # چارک اول (25امین درصد)
Q3 = df[features].quantile(0.75)  # چارک سوم (75امین درصد)

# محاسبه دامنه بین چارکی
IQR = Q3 - Q1

# شناسایی ناهنجاری‌ها
outliers = ((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR)))
outliers = outliers.any(axis=1)  # ناهنجاری در هر یک از ستون‌ها کافی است

# ذخیره نتایج
df['Anomaly_IQR'] = outliers.astype(int)  # تبدیل به 0 و 1

# نمایش ناهنجاری ها
methods = ['Anomaly_KNN', 'Anomaly_ISO', 'Anomaly_Z', 'Anomaly_IQR']
for method in methods:
    print(f"{method}: {df[method].sum()} ناهنجاری")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. KNN-Based
axes[0, 0].scatter(df['BMI'], df['Glucose'], c=df['Anomaly_KNN'], cmap='coolwarm', edgecolors='k', alpha=0.7)
axes[0, 0].set_title('KNN-Based')
axes[0, 0].set_xlabel('BMI')
axes[0, 0].set_ylabel('Glucose')
axes[0, 0].grid(True)

# 2. Isolation Forest
axes[0, 1].scatter(df['BMI'], df['Glucose'], c=df['Anomaly_ISO'], cmap='coolwarm', edgecolors='k', alpha=0.7)
axes[0, 1].set_title('Isolation Forest')
axes[0, 1].set_xlabel('BMI')
axes[0, 1].set_ylabel('Glucose')
axes[0, 1].grid(True)

# 3. Z-Score
axes[1, 0].scatter(df['BMI'], df['Glucose'], c=df['Anomaly_Z'], cmap='coolwarm', edgecolors='k', alpha=0.7)
axes[1, 0].set_title('Z-Score Method')
axes[1, 0].set_xlabel('BMI')
axes[1, 0].set_ylabel('Glucose')
axes[1, 0].grid(True)

# 4. IQR
axes[1, 1].scatter(df['BMI'], df['Glucose'], c=df['Anomaly_IQR'], cmap='coolwarm', edgecolors='k', alpha=0.7)
axes[1, 1].set_title('IQR Method')
axes[1, 1].set_xlabel('BMI')
axes[1, 1].set_ylabel('Glucose')
axes[1, 1].grid(True)

plt.tight_layout()
plt.suptitle('Visual Comparison of Anomaly Detection Methods', fontsize=16, y=1.02)
plt.show()

# ستون‌هایی که مقدار صفر در آنها معتبر نیست
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

# جایگزینی با میانگین یا میانه
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[cols_to_replace] = imputer.fit_transform(df[cols_to_replace])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = clean_df.drop('Outcome', axis=1)
y = clean_df['Outcome']

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# استانداردسازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"{name} trained successfully")

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from sklearn.model_selection import cross_val_score

results = []

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else [0]*len(X_test)

    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else None,
        'CV Score (5-fold)': cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc').mean()
    }
    results.append(metrics)

results_df = pd.DataFrame(results)
print(results_df.sort_values('ROC-AUC', ascending=False))

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# مقایسه با داده‌های اصلی (قبل از حذف ناهنجاری‌ها)
X_orig = df.drop('Outcome', axis=1)
y_orig = df['Outcome']
X_orig_scaled = scaler.transform(X_orig)

comparison_results = []
for name, model in models.items():
    y_pred_orig = model.predict(X_orig_scaled)
    metrics = {
        'Model': name,
        'Accuracy (Clean Data)': results_df[results_df['Model']==name]['Accuracy'].values[0],
        'Accuracy (Original Data)': accuracy_score(y_orig, y_pred_orig),
        'Difference': accuracy_score(y_orig, y_pred_orig) - results_df[results_df['Model']==name]['Accuracy'].values[0]
    }
    comparison_results.append(metrics)


pd.DataFrame(comparison_results).sort_values('Accuracy (Clean Data)', ascending=False)

import joblib
# ذخیره مدل Random Forest
joblib.dump(models["Random Forest"], "random_forest_model.pkl")
# ذخیره اسکیلر
joblib.dump(scaler, "scaler.pkl")

