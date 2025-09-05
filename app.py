from dash import Dash, html, dcc, Input, Output, State, ctx
from dash import dash_table
app = Dash(__name__)
application = app.server
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import base64
from io import BytesIO
from xhtml2pdf import pisa  # برای تولید PDF

# تعریف استایل‌های مشترک
BASE_STYLE = {
    'direction': 'rtl',
    'textAlign': 'right',
    'fontFamily': 'Vazir, sans-serif',
    'fontSize': '18px',  # افزایش سایز فونت پایه
    'margin': '20px',  # حاشیه 20 پیکسل برای هر باکس
    'lineHeight': '1.7'  # افزایش line space
}

HEADER_STYLE = {
    **BASE_STYLE,
    'color': '#1e3a8a',  # آبی تیره
    'padding': '15px 20px',
    'textAlign': 'center',  # وسط‌چین کردن تایتل
    'fontSize': '28px'  # افزایش سایز فونت هدر به 28px
}

INPUT_STYLE = {
    **BASE_STYLE,
    'width': '100%',
    'maxWidth': '200px',
    'padding': '10px',
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
    'padding': '15px',
    'backgroundColor': '#ffffff',
    'borderRadius': '6px',
    'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)'
}

TABLE_STYLE = {
    **BASE_STYLE,
    'width': '100%',
    'maxWidth': '800px',
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
    'borderRadius': '8px',
    'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
}

ALERT_STYLE = {
    **BASE_STYLE,
    'padding': '15px',
    'backgroundColor': '#ffe6e6',
    'border': '2px solid #ff0000',
    'borderRadius': '6px',
    'color': '#ff0000',
    'fontWeight': 'bold'
}

CONTAINER_STYLE = {
    'display': 'flex',
    'flexDirection': 'column',
    'alignItems': 'center',  # وسط‌چین کردن عناصر
    'paddingLeft': '150px',  # فاصله 150 پیکسل از چپ
    'paddingRight': '150px',  # فاصله 150 پیکسل از راست
    'width': '100%',
    'boxSizing': 'border-box'
}

# لود مدل‌ها
try:
    best_model = joblib.load("best_model_gradient_boosting.pkl")
    scaler_clean = joblib.load("scaler_clean.pkl")
    voting_model = joblib.load("voting_model_clean.pkl")
except FileNotFoundError:
    app = Dash(__name__, assets_folder='img')
    app.layout = html.P("فایل‌های مدل یافت نشد. لطفاً آن‌ها را آپلود کنید.", style={'color': '#FF0000', 'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'})
    if __name__ == '__main__':
        app.run_server(debug=True)
    exit()

# لود دیتاست
try:
    df = pd.read_csv("diabetest.csv")
except FileNotFoundError:
    app = Dash(__name__, assets_folder='img')
    app.layout = html.P("فایل دیتاست یافت نشد. لطفاً diabetest.csv را آپلود کنید.", style={'color': '#FF0000', 'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'})
    if __name__ == '__main__':
        app.run_server(debug=True)
    exit()

# تنظیم Dash با استایل خارجی
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"], assets_folder='img')

# تعریف off-canvas برای منو
offcanvas = html.Div(
    [
        html.I(
            className="fas fa-bars",
            id="open-offcanvas",
            n_clicks=0,
            style={'position': 'absolute', 'top': '15px', 'right': '60px', 'zIndex': '1000', 'fontSize': '24px', 'color': '#1e40af', 'cursor': 'pointer'}  # فقط آیکون بدون دکمه
        ),
        dbc.Offcanvas(
            [
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem("اهداف داشبورد", id="overview-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("تحلیل‌های اولیه داده‌ها", id="eda-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("تحلیل‌های تکمیلی", id="advanced-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("مدل‌های کلاسیفیکیشن و ارزیابی", id="models-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("پیش‌بینی دیابت", id="predict-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("پیشنهاد برنامه غذایی و ورزشی", id="recommendations-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'})
                    ],
                    flush=True,
                    style={'textAlign': 'right'}  # راست‌چین کردن آیتم‌های منو
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
    html.H1("داشبورد تشخیص دیابت", style=HEADER_STYLE),
    offcanvas,
    html.Div(id='page-content', style={**BASE_STYLE, **CONTAINER_STYLE})
], style=CONTAINER_STYLE)

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
    if not ctx.triggered:
        return html.P("لطفاً یک گزینه را از منو انتخاب کنید.", style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'})
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'overview-item' and overview_clicks:
        return html.Div([
            html.P("""
            اهداف اصلی:
            - تحلیل اکتشافی داده‌ها (EDA) برای درک توزیع و روابط ویژگی‌ها.
            - تشخیص و مدیریت ناهنجاری‌ها و مقادیر گمشده.
            - آموزش مدل‌های مختلف کلاسیفیکیشن و انتخاب بهترین (Gradient Boosting بعد از حذف ناهنجاری‌ها).
            - ایجاد داشبورد برای پیش‌بینی دیابت و پیشنهاد برنامه‌های شخصی‌سازی‌شده بر اساس ویژگی‌های کلیدی.
            این داشبورد با Dash ساخته شده و روی Render deploy می‌شود.
            """, style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}),
            html.Img(src='../img/img1.png', style={'width': '50%', 'margin': '20px auto', 'display': 'block'}, alt="تصویر داشبورد"),
            html.P("در صورت لود نشدن تصویر، لطفاً مطمئن شوید که فایل img1.png در پوشه img قرار دارد.", style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '16px', 'color': '#FF0000', 'margin': '10px'}),
            html.P("""
            مزایا و کاربردهای داشبورد:
            این داشبورد امکان پیش‌بینی دقیق احتمال ابتلا به دیابت را فراهم می‌کند و پیشنهادات شخصی‌سازی‌شده برای رژیم غذایی و ورزش ارائه می‌دهد. مزایای آن شامل دقت بالا در مدل‌سازی، دسترسی آسان برای کاربران، و کمک به پیشگیری از بیماری است. کاربردها عبارتند از استفاده در کلینیک‌ها برای غربالگری، ادغام با اپلیکیشن‌های سلامت، و تحقیقات پزشکی برای تحلیل داده‌های بزرگ.
            """, style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'})
        ])

    elif triggered_id == 'eda-item' and eda_clicks:
        figs = []
        for col in df.columns:
            fig_hist = px.histogram(df, x=col, nbins=20, title=f"هیستوگرام {col}")
            figs.append(dcc.Graph(figure=fig_hist, style=GRAPH_STYLE))
        fig_box = px.box(df, y=['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure'], title="باکس پلات ویژگی‌ها")
        fig_scatter = px.scatter(df, x='Glucose', y='BMI', color='Outcome', title='اسکتر پلات Glucose vs BMI', color_continuous_scale='RdBu')
        return html.Div([
            *figs,
            dcc.Graph(figure=fig_box, style=GRAPH_STYLE),
            dcc.Graph(figure=fig_scatter, style=GRAPH_STYLE)
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})

    elif triggered_id == 'advanced-item' and advanced_clicks:
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
            dcc.Graph(figure=fig_corr, style=GRAPH_STYLE),
            html.P(f"تعداد داده‌های حذف شده: **{num_removed}**", style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}),
            html.P("""
            تحلیل تکمیلی شامل:
            - شناسایی ناهنجاری‌ها با IsolationForest.
            - رگرسیون خطی برای پیش‌بینی Glucose بر اساس BMI.
            - ماتریس کورلیشن برای بررسی روابط.
            """, style={**BASE_STYLE, 'margin': '20px'})
        ]

    elif triggered_id == 'models-item' and models_clicks:
        results_df = pd.DataFrame({'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'Gradient Boosting', 'LightGBM', 'MLP'],
                                 'Accuracy': [0.75, '0.72', 0.70, 0.78, 0.80, 0.82, 0.79, 0.76]})
        fig_cm = px.imshow([[50, 10], [8, 60]], text_auto=True, color_continuous_scale='Blues', title='ماتریس درهم‌ریختگی (نمونه)')
        return [
            html.P("""
            مدل‌های تست‌شده: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, Gradient Boosting, LightGBM, MLP.
            بهترین مدل: Gradient Boosting روی داده‌های پاک‌شده (بعد از حذف ناهنجاری‌ها) با دقت بالا.
            ارزیابی شامل Accuracy, Precision, Recall, F1-Score, ROC-AUC و Cross-Validation.
            """, style={**BASE_STYLE, 'margin': '20px'}),
            dash_table.DataTable(
                data=results_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in results_df.columns],
                style_table=TABLE_STYLE,
                style_cell=TABLE_ROW_STYLE,
                style_header=TABLE_HEADER_STYLE
            ),
            dcc.Graph(figure=fig_cm, style=GRAPH_STYLE)
        ]

    elif triggered_id == 'predict-item' and predict_clicks:
        return html.Div([
            html.Label("ویژگی‌ها را وارد کنید تا مدل پیش‌بینی کند.", style={**BASE_STYLE, 'margin': '20px'}),
            *[html.Div([
                html.Label(f"{feature} (عدد {'صحیح' if feature in ['Pregnancies', 'Age'] else 'اعشاری'})", style={**BASE_STYLE, 'margin': '20px'}),
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
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})

    elif triggered_id == 'recommendations-item' and recommendations_clicks:
        return html.Div([
            html.Label("بر اساس ویژگی‌های کلیدی: Glucose, BMI, Age, Insulin, BloodPressure, Pregnancies", style={**BASE_STYLE, 'margin': '20px'}),
            *[html.Div([
                html.Label(feature, style={**BASE_STYLE, 'margin': '20px'}),
                dcc.Input(id=f'rec-input-{feature}', type='number', value=0 if feature in ['Age', 'Pregnancies'] else 0.0, step=1 if feature in ['Age', 'Pregnancies'] else 0.1, style=INPUT_STYLE)
            ]) for feature in ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure', 'Pregnancies']],
            html.Button('دریافت پیشنهاد', id='recommend-button', n_clicks=0, style=BUTTON_STYLE),
            html.Button('دانلود برنامه به صورت PDF', id='download-button', n_clicks=0, style={**BUTTON_STYLE, 'backgroundColor': '#008000', 'margin': '20px'}),
            dcc.Download(id='download-pdf'),
            html.Div(id='recommend-output', style=OUTPUT_STYLE)
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})

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

# تابع برای تولید محتوای PDF
def generate_pdf_content(glucose, bmi, age, insulin, blood_pressure, pregnancies, diet, meal_plan, sleep, exercise, walking, alerts):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'DejaVu Sans', sans-serif; direction: rtl; text-align: right; font-size: 18px; margin: 20px; line-height: 1.7; }}
            h1 {{ color: #1e3a8a; text-align: center; font-size: 24px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: right; }}
            th {{ background-color: #1e3a8a; color: white; }}
            .alert {{ color: red; font-weight: bold; border: 2px solid red; padding: 10px; margin: 20px 0; }}
            p {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>برنامه غذایی و ورزشی شخصی‌سازی‌شده</h1>
        <h2>ویژگی‌های ورودی</h2>
        <p>گلوکز: {glucose} | BMI: {bmi} | سن: {age} | انسولین: {insulin} | فشار خون: {blood_pressure} | تعداد بارداری: {pregnancies}</p>
        {'<div class="alert">' + '<br>'.join(alerts) + '</div>' if alerts else ''}
        <h2>برنامه غذایی</h2>
        <p>{diet}</p>
        <table>
            <tr><th>وعده</th><th>پیشنهاد</th><th>نکته</th></tr>
            {''.join([f'<tr><td>{row["وعده"]}</td><td>{row["پیشنهاد"]}</td><td>{row["نکته"]}</td></tr>' for _, row in meal_plan.iterrows()])}
        </table>
        <h2>خواب</h2>
        <p>{sleep}</p>
        <h2>ورزش</h2>
        <p>{exercise}</p>
        <h2>پیاده‌روی</h2>
        <p>{walking}</p>
        <p style="color: red; font-weight: bold;">نکته: این پیشنهادات عمومی هستند. برای برنامه دقیق، با پزشک مشورت کنید.</p>
    </body>
    </html>
    """

# کال‌بک برای پیشنهادات و دانلود PDF
@app.callback(
    [Output('recommend-output', 'children'),
     Output('download-pdf', 'data')],
    [Input('recommend-button', 'n_clicks'),
     Input('download-button', 'n_clicks')],
    [State(f'rec-input-{feature}', 'value') for feature in ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure', 'Pregnancies']]
)
def get_recommendations(recommend_clicks, download_clicks, glucose, bmi, age, insulin, blood_pressure, pregnancies):
    if (recommend_clicks > 0 or download_clicks > 0) and all(v is not None for v in [glucose, bmi, age, insulin, blood_pressure, pregnancies]):
        # هشدارهای پیشرفته
        alerts = []
        if glucose > 200:
            alerts.append("هشدار: سطح گلوکز شما بسیار بالاست! فوراً با پزشک مشورت کنید.")
        if insulin > 200:
            alerts.append("هشدار: سطح انسولین غیرطبیعی است. لطفاً با متخصص غدد مشورت کنید.")
        if blood_pressure > 140:
            alerts.append("هشدار: فشار خون بالا! به پزشک قلب مراجعه کنید.")
        if bmi > 30:
            alerts.append("هشدار: BMI شما نشان‌دهنده چاقی است. کاهش وزن ضروری است.")
        if pregnancies > 6:
            alerts.append("هشدار: تعداد بالای بارداری ممکن است خطر دیابت را افزایش دهد. با پزشک مشورت کنید.")

        # محاسبه کالری روزانه تخمینی
        bmr = 10 * bmi + 6.25 * 170 - 5 * age + 5  # فرض قد متوسط 170cm
        daily_calories = bmr * 1.2 if bmi > 25 else bmr * 1.5

        # رژیم غذایی
        diet = "برنامه غذایی شخصی‌سازی‌شده: "
        if glucose > 140:
            diet += "تمرکز روی غذاهای با شاخص گلیسمیک پایین (GI < 55). اجتناب از شکر و کربوهیدرات‌های ساده. "
        if bmi > 25:
            diet += f"کالری روزانه پیشنهادی: {int(daily_calories)} کالری برای کاهش وزن. "
            diet += "اجتناب از فست‌فود و نوشابه‌ها. "
        else:
            diet += f"کالری روزانه پیشنهادی: {int(daily_calories)} کالری برای حفظ وزن. تعادل کربوهیدرات (45-65%)، پروتئین (10-35%) و چربی (20-35%). "
        if pregnancies > 0:
            diet += "برای زنان باردار: افزایش مصرف فولات و آهن از سبزیجات برگ‌دار و گوشت بدون چربی. "

        # جدول برنامه غذایی
        meal_plan = pd.DataFrame({
            'وعده': ['صبحانه', 'میان‌وعده', 'ناهار', 'میان‌وعده', 'شام'],
            'پیشنهاد': [
                'تخم‌مرغ آب‌پز با سبزیجات و نان جو کامل (300 کالری)',
                'یک سیب متوسط با بادام (200 کالری)',
                'سالاد مرغ گریل با کینوا و سبزیجات (500 کالری)',
                'ماست کم‌چرب با توت‌ها (150 کالری)',
                'ماهی سالمون کبابی با بروکلی و برنج قهوه‌ای (400 کالری)'
            ],
            'نکته': [
                'پروتئین بالا برای کنترل قند خون',
                'فیبر بالا برای احساس سیری',
                'تعادل پروتئین و کربوهیدرات پیچیده',
                'کم‌شکر برای جلوگیری از پیک انسولین',
                'اسیدهای چرب امگا-3 برای سلامت قلب'
            ]
        })
        if glucose > 140 or insulin > 100:
            meal_plan['پیشنهاد'] = meal_plan['پیشنهاد'].apply(lambda x: x + ' (کم GI)')

        diet_table = dash_table.DataTable(
            data=meal_plan.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in meal_plan.columns],
            style_table=TABLE_STYLE,
            style_cell=TABLE_ROW_STYLE,
            style_header=TABLE_HEADER_STYLE
        )

        # خواب
        sleep = "خواب کافی: "
        if age < 30:
            sleep += "8-9 ساعت در شب. نکته: اتاق تاریک و خنک برای کیفیت بهتر خواب."
        elif age < 50:
            sleep += "7-8 ساعت. نکته: اجتناب از کافئین بعد از ظهر."
        else:
            sleep += "6-7 ساعت، با چرت کوتاه 20 دقیقه‌ای روزانه. نکته: روتین خواب منظم."

        # ورزش
        exercise = "برنامه ورزشی شخصی: "
        if bmi > 25 or insulin > 100:
            exercise += "45-60 دقیقه روزانه: 30 دقیقه ایروبیک (دویدن یا دوچرخه) + 15 دقیقه وزنه‌برداری سبک. "
            exercise += "هدف: سوزاندن 300-500 کالری در جلسه."
        else:
            exercise += "30 دقیقه متوسط: یوگا، شنا یا پیاده‌روی سریع. "
            exercise += "هدف: حفظ تناسب اندام و کنترل قند."
        if blood_pressure > 120:
            exercise += "تمرکز روی ورزش‌های آرام‌بخش مثل یوگا برای کاهش فشار خون."

        # پیاده‌روی
        walking = "پیاده‌روی: "
        if bmi > 25:
            walking += "حداقل 10000 قدم روزانه (تقریباً 8 کیلومتر، سوزاندن ~400 کالری). "
        else:
            walking += "حداقل 5000 قدم (سوزاندن ~200 کالری). "
        walking += "نکته: استفاده از اپ‌هایی مثل Google Fit برای پیگیری و چالش‌های روزانه."

        # نمودار توزیع کالری
        fig_calories = px.pie(values=[daily_calories * 0.5, daily_calories * 0.3, daily_calories * 0.2], 
                              names=['کربوهیدرات', 'پروتئین', 'چربی'], 
                              title='توزیع پیشنهادی کالری روزانه',
                              color_discrete_sequence=px.colors.sequential.Blues)

        # تولید محتوای PDF
        pdf_content = generate_pdf_content(glucose, bmi, age, insulin, blood_pressure, pregnancies, diet, meal_plan, sleep, exercise, walking, alerts)
        output = BytesIO()
        pisa.CreatePDF(pdf_content, dest=output)
        pdf_data = output.getvalue()
        output.close()

        # خروجی رندر شده در داشبورد
        output_children = [
            html.Div([html.P(alert, style=ALERT_STYLE) for alert in alerts]) if alerts else html.Div(),
            html.P(diet, style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}),
            diet_table,
            html.P(sleep, style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}),
            html.P(exercise, style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}),
            html.P(walking, style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}),
            dcc.Graph(figure=fig_calories, style=GRAPH_STYLE),
            html.P("نکته کلی: این پیشنهادات عمومی هستند. برای برنامه دقیق، با پزشک مشورت کنید.", style={'direction': 'rtl', 'text-align': 'right', 'color': '#FF0000', 'fontSize': '18px', 'margin': '20px'})
        ]

        # اگر دکمه دانلود کلیک شده باشد
        if ctx.triggered_id == 'download-button':
            return output_children, dcc.send_bytes(pdf_data, "برنامه_غذایی_و_ورزشی.pdf")
        return output_children, None

    return html.P("لطفاً مقادیر را وارد کنید.", style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}), None

if __name__ == '__main__':
    app.run_server(debug=True)
    
# برای WSGI (مثل waitress)
application = app.server  # این خط شیء WSGI رو از Dash می‌سازه
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8000)

