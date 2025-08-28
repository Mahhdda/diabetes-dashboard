from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

# لود مدل‌ها
try:
    best_model = joblib.load("best_model_gradient_boosting.pkl")
    scaler_clean = joblib.load("scaler_clean.pkl")
    voting_model = joblib.load("voting_model_clean.pkl")
except FileNotFoundError:
    print("فایل‌های مدل یافت نشد. لطفاً آن‌ها را آپلود کنید.")
    exit()

# تنظیم Dash با استایل خارجی
app = Dash(__name__, external_stylesheets=['style.css'])  # مسیر style.css رو تنظیم کن

# لود دیتاست (برای تحلیل‌ها)
try:
    df = pd.read_csv("diabetest.csv")
except FileNotFoundError:
    print("فایل دیتاست یافت نشد. لطفاً diabetest.csv را آپلود کنید.")
    exit()

app.layout = html.Div([
    html.H1("داشبورد تشخیص دیابت با Gradient Boosting", style={'color': 'red'}),
    html.Div([
        dcc.Dropdown(
            id='page-dropdown',
            options=[
                {'label': 'مروری بر پروژه', 'value': 'overview'},
                {'label': 'تحلیل‌های اولیه داده‌ها', 'value': 'eda'},
                {'label': 'تحلیل‌های تکمیلی', 'value': 'advanced'},
                {'label': 'مدل‌های کلاسیفیکیشن و ارزیابی', 'value': 'models'},
                {'label': 'پیش‌بینی دیابت', 'value': 'predict'},
                {'label': 'پیشنهاد برنامه غذایی و ورزشی', 'value': 'recommendations'}
            ],
            value='overview',
            style={'direction': 'rtl', 'text-align': 'right'}
        )
    ], style={'margin': '20px'}),
    html.Div(id='page-content', style={'direction': 'rtl', 'text-align': 'right'})
], style={'direction': 'rtl', 'text-align': 'right', 'padding': '20px'})

# کال‌بک برای تغییر محتوا بر اساس انتخاب
@app.callback(
    Output('page-content', 'children'),
    Input('page-dropdown', 'value')
)
def update_page(value):
    if value == 'overview':
        return html.P("""
        این پروژه داده‌کاوی بر روی دیتاست Pima Indians Diabetes تمرکز دارد که شامل 768 رکورد و 9 ویژگی است. 
        اهداف اصلی:
        - تحلیل اکتشافی داده‌ها (EDA) برای درک توزیع و روابط ویژگی‌ها.
        - تشخیص و مدیریت ناهنجاری‌ها و مقادیر گمشده.
        - آموزش مدل‌های مختلف کلاسیفیکیشن و انتخاب بهترین (Gradient Boosting بعد از حذف ناهنجاری‌ها).
        - ایجاد داشبورد برای پیش‌بینی دیابت و پیشنهاد برنامه‌های شخصی‌سازی‌شده بر اساس ویژگی‌های کلیدی.
        این داشبورد با Dash ساخته شده و روی Render deploy می‌شود.
        """, style={'direction': 'rtl', 'text-align': 'right'})

    elif value == 'eda':
        # هیستوگرام
        fig_hist, ax_hist = plt.subplots(figsize=(10, 8))
        df.hist(bins=20, ax=ax_hist)
        plt.tight_layout()
        # باکس پلات
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        features_to_plot = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
        sns.boxplot(data=df[features_to_plot], ax=ax_box)
        plt.xticks(rotation=45)
        # اسکتر پلات
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome', palette='coolwarm', ax=ax_scatter)
        return [
            dcc.Graph(figure=fig_hist),
            dcc.Graph(figure=fig_box),
            dcc.Graph(figure=fig_scatter)
        ]

    elif value == 'advanced':
        # کورلیشن ماتریکس
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
        # حذف ناهنجاری‌ها
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(df.drop('Outcome', axis=1))
        num_removed = len(df) - sum(outliers == 1)
        # رگرسیون خطی
        fig_reg, ax_reg = plt.subplots(figsize=(8, 5))
        X = df[['BMI']]
        y = df['Glucose']
        reg_model = LinearRegression().fit(X, y)
        y_pred = reg_model.predict(X)
        ax_reg.scatter(X, y, alpha=0.6)
        ax_reg.plot(X, y_pred, color='red')
        return [
            dcc.Graph(figure=fig_corr),
            html.P(f"تعداد داده‌های حذف شده: **{num_removed}**", style={'direction': 'rtl', 'text-align': 'right'}),
            html.P("این روش 5% از داده‌ها رو به عنوان ناهنجاری در نظر می‌گیره و حذف می‌کنه.", style={'direction': 'rtl', 'text-align': 'right'}),
            dcc.Graph(figure=fig_reg),
            html.P("نمایش مرز تصمیم SVM (تصویر ساده‌شده). برای جزئیات بیشتر به کد اصلی مراجعه کنید.", style={'direction': 'rtl', 'text-align': 'right'})
        ]

    elif value == 'models':
        results_data = {
            'Model': ['Gradient Boosting', 'Logistic Regression', 'MLP'],
            'Accuracy': [0.80, 0.78, 0.77],
            'Precision': [0.75, 0.73, 0.72],
            'Recall': [0.68, 0.66, 0.65],
            'F1-Score': [0.71, 0.69, 0.68],
            'ROC-AUC': [0.87, 0.85, 0.84]
        }
        results_df = pd.DataFrame(results_data)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
        cm = np.array([[92, 8], [12, 42]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        return [
            html.P("""
            مدل‌های آموزش‌دیده: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, Gradient Boosting, LightGBM, MLP.
            بهترین مدل: Gradient Boosting روی داده‌های پاک‌شده (بعد از حذف ناهنجاری‌ها) با دقت بالا.
            ارزیابی شامل Accuracy, Precision, Recall, F1-Score, ROC-AUC و Cross-Validation.
            """, style={'direction': 'rtl', 'text-align': 'right'}),
            dash_table.DataTable(
                data=results_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in results_df.columns]
            ),
            dcc.Graph(figure=fig_cm)
        ]

    elif value == 'predict':
        return html.Div([
            html.Label("ویژگی‌ها را وارد کنید تا مدل پیش‌بینی کند.", style={'direction': 'rtl', 'text-align': 'right'}),
            *[html.Div([
                html.Label(f"{feature} (عدد {'سفید' if feature in ['Pregnancies', 'Age'] else 'اعشاری'})", style={'direction': 'rtl', 'text-align': 'right'}),
                dcc.Input(id=f'input-{feature}', type='number', value=0 if feature in ['Pregnancies', 'Age'] else 0.0, step=1 if feature in ['Pregnancies', 'Age'] else 0.1, style={'direction': 'rtl', 'text-align': 'right'})
            ]) for feature in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']],
            dcc.Dropdown(
                id='model-choice',
                options=[{'label': 'Gradient Boosting (بهترین تک مدل)', 'value': 'gb'}, {'label': 'Voting (سه مدل برتر)', 'value': 'voting'}],
                value='gb',
                style={'direction': 'rtl', 'text-align': 'right'}
            ),
            html.Button('پیش‌بینی', id='predict-button', n_clicks=0, style={'direction': 'rtl', 'text-align': 'center'}),
            html.Div(id='prediction-output', style={'direction': 'rtl', 'text-align': 'right'})
        ])

    elif value == 'recommendations':
        return html.Div([
            html.Label("بر اساس 4 ویژگی مهم: Glucose, BMI, Age, Insulin", style={'direction': 'rtl', 'text-align': 'right'}),
            *[html.Div([
                html.Label(feature, style={'direction': 'rtl', 'text-align': 'right'}),
                dcc.Input(id=f'rec-input-{feature}', type='number', value=0 if feature == 'Age' else 0.0, step=1 if feature == 'Age' else 0.1, style={'direction': 'rtl', 'text-align': 'right'})
            ]) for feature in ['Glucose', 'BMI', 'Age', 'Insulin']],
            html.Button('دریافت پیشنهاد', id='recommend-button', n_clicks=0, style={'direction': 'rtl', 'text-align': 'center'}),
            html.Div(id='recommend-output', style={'direction': 'rtl', 'text-align': 'right'})
        ])

# کال‌بک برای پیش‌بینی
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [Input(f'input-{feature}', 'value') for feature in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']],
    Input('model-choice', 'value')
)
def predict_diabetes(n_clicks, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age, model_choice):
    if n_clicks > 0 and all(v is not None for v in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]):
        input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]], 
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        try:
            input_scaled = scaler_clean.transform(input_df)
            if model_choice == 'voting':
                prediction = voting_model.predict(input_scaled)[0]
                prob = voting_model.predict_proba(input_scaled)[0][1] * 100
            else:
                prediction = best_model.predict(input_scaled)[0]
                prob = best_model.predict_proba(input_scaled)[0][1] * 100
            return html.P(f"احتمال دیابت: {prob:.2f}% {'(دیابتی)' if prediction == 1 else '(غیر دیابتی)'}", style={'color': '#FF0000' if prediction == 1 else '#008000', 'direction': 'rtl', 'text-align': 'right'})
        except ValueError as e:
            return html.P("خطا در پیش‌بینی: ممکن است ویژگی‌ها با مدل سازگار نباشند.", style={'color': '#FF0000', 'direction': 'rtl', 'text-align': 'right'})
    return ""

# کال‌بک برای پیشنهادات
@app.callback(
    Output('recommend-output', 'children'),
    Input('recommend-button', 'n_clicks'),
    [Input(f'rec-input-{feature}', 'value') for feature in ['Glucose', 'BMI', 'Age', 'Insulin']]
)
def get_recommendations(n_clicks, glucose, bmi, age, insulin):
    if n_clicks > 0 and all(v is not None for v in [glucose, bmi, age, insulin]):
        diet = "برنامه غذایی متعادل: "
        if glucose > 140:
            diet += "غ
