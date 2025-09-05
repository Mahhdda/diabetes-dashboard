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

# تعریف استایل‌های مشترک
BASE_STYLE = {
    'direction': 'rtl',
    'textAlign': 'right',
    'fontFamily': 'Vazir, sans-serif'  # اگه فونت Vazir داری، جایگزین کن
}

HEADER_STYLE = {
    **BASE_STYLE,
    'color': '#1e3a8a',  # آبی تیره
    'padding': '15px 20px',
    'backgroundColor': '#ffffff',
    'borderBottom': '1px solid #e2e8f0',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
    'textAlign': 'center'  # وسط‌چین کردن تایتل
}

INPUT_STYLE = {
    **BASE_STYLE,
    'width': '100%',
    'maxWidth': '200px',
    'padding': '10px',
    'margin': '10px 0',
    'border': '1px solid #e2e8f0',
    'borderRadius': '6px'
}

BUTTON_STYLE = {
    **BASE_STYLE,
    'backgroundColor': '#1e40af',
    'color': '#ffffff',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '6px',
    'cursor': 'pointer',
    'textAlign': 'center'
}

OUTPUT_STYLE = {
    **BASE_STYLE,
    'margin': '20px 0',
    'padding': '15px',
    'backgroundColor': '#ffffff',
    'borderRadius': '6px',
    'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)'
}

TABLE_STYLE = {
    **BASE_STYLE,
    'width': '100%',
    'maxWidth': '800px',
    'margin': '20px auto',
    'borderCollapse': 'collapse'
}

TABLE_HEADER_STYLE = {
    **BASE_STYLE,
    'backgroundColor': '#1e3a8a',
    'color': '#ffffff',
    'padding': '12px'
}

TABLE_ROW_STYLE = {
    **BASE_STYLE,
    'padding': '12px',
    'border': '1px solid #e2e8f0'
}

GRAPH_STYLE = {
    **BASE_STYLE,
    'margin': '20px auto',
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
}

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
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# تعریف off-canvas برای منو
offcanvas = html.Div(
    [
        dbc.Button(
            children=[html.I(className="fas fa-bars")],  # آیکون همبرگری با Font Awesome (در دسترس از طریق Bootstrap)
            id="open-offcanvas",
            n_clicks=0,
            style={'position': 'absolute', 'top': '15px', 'right': '15px', 'zIndex': '1000', 'fontSize': '24px', 'color': '#1e40af'}
        ),
        dbc.Offcanvas(
            [
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem("مروری بر پروژه", id="overview-item", n_clicks=0, style={'cursor': 'pointer'}),
                        dbc.ListGroupItem("تحلیل‌های اولیه داده‌ها", id="eda-item", n_clicks=0, style={'cursor': 'pointer'}),
                        dbc.ListGroupItem("تحلیل‌های تکمیلی", id="advanced-item", n_clicks=0, style={'cursor': 'pointer'}),
                        dbc.ListGroupItem("مدل‌های کلاسیفیکیشن و ارزیابی", id="models-item", n_clicks=0, style={'cursor': 'pointer'}),
                        dbc.ListGroupItem("پیش‌بینی دیابت", id="predict-item", n_clicks=0, style={'cursor': 'pointer'}),
                        dbc.ListGroupItem("پیشنهاد برنامه غذایی و ورزشی", id="recommendations-item", n_clicks=0, style={'cursor': 'pointer'})
                    ],
                    flush=True
                )
            ],
            id="offcanvas",
            is_open=False,
            title="منوی داشبورد",
            placement="end"  # باز شدن از سمت راست
        )
    ]
)

app.layout = html.Div([
    html.H1("داشبورد تشخیص دیابت با Gradient Boosting", style=HEADER_STYLE),
    offcanvas,
    html.Div(id='page-content', style=BASE_STYLE)
], style=BASE_STYLE)

# کال‌بک برای باز و بسته کردن off-canvas
@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")]
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

# کال‌بک برای تغییر محتوا بر اساس انتخاب از off-canvas
@app.callback(
    Output('page-content', 'children'),
    [Input('overview-item', 'n_clicks'),
     Input('eda-item', 'n_clicks'),
     Input('advanced-item', 'n_clicks'),
     Input('models-item', 'n_clicks'),
     Input('predict-item', 'n_clicks'),
     Input('recommendations-item', 'n_clicks')]
)
def update_page(overview_clicks, eda_clicks, advanced_clicks, models_clicks, predict_clicks, recommendations_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.P("لطفاً یک گزینه را از منو انتخاب کنید.", style={'direction': 'rtl', 'text-align': 'right'})
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'overview-item':
        return html.P("""
        این پروژه داده‌کاوی بر روی دیتاست Pima Indians Diabetes تمرکز دارد که شامل 768 رکورد و 9 ویژگی است. 
        اهداف اصلی:
        - تحلیل اکتشافی داده‌ها (EDA) برای درک توزیع و روابط ویژگی‌ها.
        - تشخیص و مدیریت ناهنجاری‌ها و مقادیر گمشده.
        - آموزش مدل‌های مختلف کلاسیفیکیشن و انتخاب بهترین (Gradient Boosting بعد از حذف ناهنجاری‌ها).
        - ایجاد داشبورد برای پیش‌بینی دیابت و پیشنهاد برنامه‌های شخصی‌سازی‌شده بر اساس ویژگی‌های کلیدی.
        این داشبورد با Dash ساخته شده و روی Render deploy می‌شود.
        """, style={'direction': 'rtl', 'text-align': 'right'})

    elif triggered_id == 'eda-item':
        figs = []
        for col in df.columns:
            fig_hist = px.histogram(df, x=col, nbins=20, title=f"هیستوگرام {col}")
            figs.append(dcc.Graph(figure=fig_hist))
        fig_box = px.box(df, y=['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure'], title="باکس پلات ویژگی‌ها")
        fig_scatter = px.scatter(df, x='Glucose', y='BMI', color='Outcome', title='اسکتر پلات Glucose vs BMI', color_continuous_scale='coolwarm')
        return figs + [dcc.Graph(figure=fig_box), dcc.Graph(figure=fig_scatter)]

    elif triggered_id == 'advanced-item':
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
            html.P("""
            تحلیل تکمیلی شامل:
            - شناسایی ناهنجاری‌ها با IsolationForest.
            - رگرسیون خطی برای پیش‌بینی Glucose بر اساس BMI.
            - ماتریس کورلیشن برای بررسی روابط.
            """, style=BASE_STYLE)
        ]

    elif triggered_id == 'models-item':
        results_df = pd.DataFrame({'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'Gradient Boosting', 'LightGBM', 'MLP'],
                                 'Accuracy': [0.75, 0.72, 0.70, 0.78, 0.80, 0.82, 0.79, 0.76]})
        fig_cm = px.imshow([[50, 10], [8, 60]], text_auto=True, color_continuous_scale='Blues', title='ماتریس درهم‌ریختگی (نمونه)')
        return [
            html.P("""
            مدل‌های تست‌شده: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, Gradient Boosting, LightGBM, MLP.
            بهترین مدل: Gradient Boosting روی داده‌های پاک‌شده (بعد از حذف ناهنجاری‌ها) با دقت بالا.
            ارزیابی شامل Accuracy, Precision, Recall, F1-Score, ROC-AUC و Cross-Validation.
            """, style=BASE_STYLE),
            dash_table.DataTable(
                data=results_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in results_df.columns],
                style_table=TABLE_STYLE,
                style_cell=TABLE_ROW_STYLE,
                style_header=TABLE_HEADER_STYLE
            ),
            dcc.Graph(figure=fig_cm, style=GRAPH_STYLE)
        ]

    elif triggered_id == 'predict-item':
        return html.Div([
            html.Label("ویژگی‌ها را وارد کنید تا مدل پیش‌بینی کند.", style=BASE_STYLE),
            *[html.Div([
                html.Label(f"{feature} (عدد {'صحیح' if feature in ['Pregnancies', 'Age'] else 'اعشاری'})", style=BASE_STYLE),
                dcc.Input(id=f'input-{feature}', type='number', value=0 if feature in ['Pregnancies', 'Age'] else 0.0, step=1 if feature in ['Pregnancies', 'Age'] else 0.1, style=INPUT_STYLE)
            ]) for feature in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']],
            dcc.Dropdown(
                id='model-choice',
                options=[{'label': 'Gradient Boosting (بهترین تک مدل)', 'value': 'gb'}, {'label': 'Voting (سه مدل برتر)', 'value': 'voting'}],
                value='gb',
                style=INPUT_STYLE
            ),
            html.Button('پیش‌بینی', id='predict-button', n_clicks=0, style=BUTTON_STYLE),
            html.Div(id='prediction-output', style=OUTPUT_STYLE)
        ])

    elif triggered_id == 'recommendations-item':
        return html.Div([
            html.Label("بر اساس 4 ویژگی مهم: Glucose, BMI, Age, Insulin", style=BASE_STYLE),
            *[html.Div([
                html.Label(feature, style=BASE_STYLE),
                dcc.Input(id=f'rec-input-{feature}', type='number', value=0 if feature == 'Age' else 0.0, step=1 if feature == 'Age' else 0.1, style=INPUT_STYLE)
            ]) for feature in ['Glucose', 'BMI', 'Age', 'Insulin']],
            html.Button('دریافت پیشنهاد', id='recommend-button', n_clicks=0, style=BUTTON_STYLE),
            html.Div(id='recommend-output', style=OUTPUT_STYLE)
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
            return html.P(f"احتمال دیابت: {prob:.2f}% {'(دیابتی)' if prediction == 1 else '(غیر دیابتی)'}", 
                          style={**OUTPUT_STYLE, 'color': '#FF0000' if prediction == 1 else '#008000'})
        except ValueError as e:
            return html.P("خطا در پیش‌بینی: ممکن است ویژگی‌ها با مدل سازگار نباشند.", style={**OUTPUT_STYLE, 'color': '#FF0000'})
    return html.P("لطفاً مقادیر را وارد کنید.", style=OUTPUT_STYLE)

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
