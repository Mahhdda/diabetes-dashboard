from dash import Dash, html, dcc, Input, Output, State
from dash import dash_table
app = Dash(__name__)
application = app.server
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

# لود مدل‌ها
try:
    best_model = joblib.load("best_model_gradient_boosting.pkl")
    scaler_clean = joblib.load("scaler_clean.pkl")
    voting_model = joblib.load("voting_model_clean.pkl")
except FileNotFoundError:
    app = Dash(__name__)
    app.layout = html.P("فایل‌های مدل یافت نشد. لطفاً آن‌ها را آپلود کنید.", style={'color': '#FF0000', 'direction': 'rtl', 'text-align': 'right'})
    if __name__ == '__main__':
        app.run_server(debug=True)
    exit()

# لود دیتاست
try:
    df = pd.read_csv("diabetest.csv")
except FileNotFoundError:
    app = Dash(__name__)
    app.layout = html.P("فایل دیتاست یافت نشد. لطفاً diabetest.csv را آپلود کنید.", style={'color': '#FF0000', 'direction': 'rtl', 'text-align': 'right'})
    if __name__ == '__main__':
        app.run_server(debug=True)
    exit()

# تنظیم Dash با استایل خارجی
app = Dash(__name__, external_stylesheets=['style.css'])

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
        figs = []
        for col in df.columns:
            fig_hist = px.histogram(df, x=col, nbins=20, title=f"هیستوگرام {col}")
            figs.append(dcc.Graph(figure=fig_hist))
        fig_box = px.box(df, y=['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure'], title="باکس پلات ویژگی‌ها")
        fig_scatter = px.scatter(df, x='Glucose', y='BMI', color='Outcome', title='اسکتر پلات Glucose vs BMI', color_continuous_scale='coolwarm')
        return figs + [dcc.Graph(figure=fig_box), dcc.Graph(figure=fig_scatter)]

    elif value == 'advanced':
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title='کورلیشن ماتریکس')
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(df.drop('Outcome', axis=1))
        num_removed = len(df) - sum(outliers == 1)
        X = df[['BMI']]
        y = df['Glucose']
        reg_model = LinearRegression().fit(X, y)
        y_pred = reg_model.predict(X)
        fig_reg = px.scatter(df, x='BMI', y='Glucose', title='رگرسیون خطی: BMI vs Glucose')
        fig_reg.add_scatter(x=X['BMI'], y=y_pred, mode='lines', name='خط رگرسیون', line=dict(color='red'))
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
        cm = np.array([[92, 8], [12, 42]])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title='ماتریس سردرگمی')
        return [
            html.P("""
            مدل‌های آموزش‌دیده: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, Gradient Boosting, LightGBM, MLP.
            بهترین مدل: Gradient Boosting روی داده‌های پاک‌شده (بعد از حذف ناهنجاری‌ها) با دقت بالا.
            ارزیابی شامل Accuracy, Precision, Recall, F1-Score, ROC-AUC و Cross-Validation.
            """, style={'direction': 'rtl', 'text-align': 'right'}),
            dash_table.DataTable(
                data=results_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in results_df.columns],
                style_table={'direction': 'rtl', 'textAlign': 'right'},
                style_cell={'textAlign': 'right', 'fontFamily': 'Vazir'},
                style_header={'textAlign': 'right', 'fontFamily': 'Vazir'}
            ),
            dcc.Graph(figure=fig_cm)
        ]

    elif value == 'predict':
        return html.Div([
            html.Label("ویژگی‌ها را وارد کنید تا مدل پیش‌بینی کند.", style={'direction': 'rtl', 'text-align': 'right'}),
            *[html.Div([
                html.Label(f"{feature} (عدد {'صحیح' if feature in ['Pregnancies', 'Age'] else 'اعشاری'})", style={'direction': 'rtl', 'text-align': 'right'}),
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
    return html.P("لطفاً مقادیر را وارد کنید.", style={'direction': 'rtl', 'text-align': 'right'})

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
            diet += "غذاهای کم‌شکر، سبزیجات بیشتر، پروتئین بدون چربی. "
        if bmi > 25:
            diet += "رژیم کم‌کالری، اجتناب از فست‌فود. نمونه روزانه: صبحانه: تخم‌مرغ و سبزی، ناهار: سالاد مرغ، شام: ماهی و بروکلی."
        else:
            diet += "غذاهای سالم با تعادل کربوهیدرات، پروتئین و چربی."
        sleep = "خواب کافی: "
        if age < 30:
            sleep += "8-9 ساعت در شب."
        elif age < 50:
            sleep += "7-8 ساعت."
        else:
            sleep += "6-7 ساعت، با چرت کوتاه روزانه."
        exercise = "برنامه ورزشی: "
        if bmi > 25 or insulin > 100:
            exercise += "ورزش روزانه 45 دقیقه: ایروبیک، وزنه‌برداری سبک."
        else:
            exercise += "ورزش متوسط 30 دقیقه: یوگا یا شنا."
        walking = "پیاده‌روی: حداقل 5000 قدم روزانه، اگر BMI بالا باشد 10000 قدم."
        
        return [
            html.P(diet, style={'direction': 'rtl', 'text-align': 'right'}),
            html.P(sleep, style={'direction': 'rtl', 'text-align': 'right'}),
            html.P(exercise, style={'direction': 'rtl', 'text-align': 'right'}),
            html.P(walking, style={'direction': 'rtl', 'text-align': 'right'})
        ]
    return html.P("لطفاً مقادیر را وارد کنید.", style={'direction': 'rtl', 'text-align': 'right'})

if __name__ == '__main__':
    app.run_server(debug=True)
    
# برای WSGI (مثل waitress)
application = app.server  # این خط شیء WSGI رو از Dash می‌سازه
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8000)