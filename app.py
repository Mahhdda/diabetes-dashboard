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
    'border-radius': '27px',  # حاشیه 20 پیکسل برای هر باکس
    'lineHeight': '1.7'  # افزایش line space
}

HEADER_STYLE = {
    **BASE_STYLE,
    'color': '#1e3a8a',  # آبی تیره
    'textAlign': 'center',  # وسط‌چین کردن تایتل
    'fontSize': '30px'  # افزایش سایز فونت هدر به 28px
}

# استایل جدید برای هدر اصلی
MAIN_HEADER_STYLE = {
    'background': 'linear-gradient(to bottom, #ebfdff, #ffffff)',
    'padding': '20px',
    'display': 'flex',
    'justifyContent': 'space-between',
    'alignItems': 'center',
    'width': '100%',
    'position': 'sticky',
    'top': '0',
    'margin-top': '20px',
    'margin-left': '20px',
    'margin-right': '20px',
    'textAlign': 'center',
    'zIndex': '1000',
    'border-radius': '27px'
}

# استایل برای بنر
BANNER_STYLE = {
    'background': 'linear-gradient(to bottom, #9ed5da, #277d8d)',
    'padding': '40px',
    'width': '100%',
    'textAlign': 'center',
    'color': '#ffffff',
    'border-radius': '27px',
    'margin-top': '20px'
}

# تعریف SVG به صورت رشته
SVG_CONTENT = """<?xml version="1.0" encoding="iso-8859-1"?>
<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
     viewBox="0 0 512 512" xml:space="preserve">
<circle style="fill:#0F7986;" cx="255.492" cy="256" r="255.492"/>
<path style="fill:#D9DADA;" d="M75.505,40.601h50.344v188.451c0,13.845-11.326,25.171-25.171,25.171l0,0
    c-13.845,0-25.171-11.326-25.171-25.171V40.601H75.505z"/>
<path style="fill:#EEF3F7;" d="M75.505,40.601h46.64v190.405c0,8-3.608,15.199-9.275,20.048c-3.619,2.017-7.776,3.17-12.191,3.17
    l0,0c-13.845,0-25.174-11.326-25.174-25.174V40.601z"/>
<path style="fill:#FF5B62;" d="M80.71,102.508V229.16c0,11.052,9.009,20.131,19.969,20.131c10.963,0,19.969-9.08,19.969-20.131
    V102.508H80.71z"/>
<path style="fill:#EEF3F7;" d="M70.831,33.629h59.696c3.899,0,7.059,3.16,7.059,7.058l0,0c0,3.898-3.161,7.058-7.059,7.058H70.831
    c-3.899,0-7.059-3.16-7.059-7.058l0,0C63.771,36.788,66.933,33.629,70.831,33.629z"/>
<path style="fill:#CCD8DB;" d="M137.587,40.688c0,3.88-3.175,7.057-7.057,7.057H70.828c-3.88,0-7.057-3.175-7.057-7.057H137.587z"/>
<path style="fill:#FEFEFE;" d="M95.764,119.378c2.521,0,4.567,2.046,4.567,4.567s-2.046,4.567-4.567,4.567
    c-2.524,0-4.57-2.046-4.57-4.567C91.195,121.421,93.241,119.378,95.764,119.378z M98.196,197.036c0.671,0,1.217,0.544,1.217,1.217
    c0,0.671-0.544,1.217-1.217,1.217c-0.674,0-1.217-0.544-1.217-1.217C96.979,197.579,97.523,197.036,98.196,197.036z M94.848,176.021
    c2.521,0,4.567,2.046,4.567,4.567c0,2.524-2.046,4.567-4.567,4.567c-2.524,0-4.567-2.046-4.567-4.567
    C90.28,178.067,92.326,176.021,94.848,176.021z M107.03,155.922c4.036,0,7.31,3.274,7.31,7.307c0,4.036-3.274,7.31-7.31,7.31
    c-4.036,0-7.307-3.274-7.307-7.31C99.722,159.196,102.994,155.922,107.03,155.922z"/>
<rect x="168.591" y="76.86" style="fill:#21D0C3;" width="281.002" height="364.626"/>
<polygon style="fill:#DCE3DB;" points="178.699,89.473 430.933,89.473 439.483,430.51 187.249,430.51 "/>
<rect x="182.906" y="89.471" style="fill:#FFFFFF;" width="248.021" height="332.651"/>
<path style="fill:#666666;" d="M276.541,67.993h67.926c4.891,0,8.893,4.002,8.893,8.893v29.195h-85.713V76.888
    C267.647,71.996,271.65,67.993,276.541,67.993z"/>
<path style="fill:#B0B0B0;" d="M265.808,103.876h89.389c1.891,0,3.423,1.532,3.423,3.423v0.003c0,1.891-1.532,3.423-3.423,3.423
    h-89.389c-1.891,0-3.423-1.532-3.423-3.423v-0.003C262.385,105.409,263.918,103.876,265.808,103.876z"/>
<path style="fill:#FEFEFE;" d="M301.571,59.131l-7.23,44.745h-3.72l7.209-44.636c-2.051-2.843-3.094-6.253-3.094-9.6
    c0-2.145,0.428-4.271,1.291-6.221c0.873-1.967,2.189-3.749,3.959-5.176c5.628-4.541,15.384-4.546,21.012-0.003
    c1.774,1.431,3.094,3.218,3.971,5.192c0.865,1.954,1.296,4.092,1.296,6.242c0,3.336-1.035,6.731-3.078,9.566l7.209,44.636h-3.72
    l-7.23-44.745c-0.108-0.567,0.06-1.124,0.407-1.531c1.829-2.286,2.754-5.126,2.754-7.922c0-1.657-0.327-3.289-0.982-4.768
    c-0.647-1.458-1.618-2.775-2.911-3.82c-4.277-3.452-12.16-3.447-16.434,0.003c-1.291,1.04-2.257,2.352-2.9,3.804
    c-0.65,1.473-0.974,3.099-0.974,4.747c0,2.824,0.946,5.696,2.803,8c0.35,0.433,0.468,0.98,0.371,1.486l0,0L301.571,59.131
    L301.571,59.131z"/>
<rect x="417.831" y="162.917" transform="matrix(0.8067 0.591 -0.591 0.8067 210.5895 -211.475)" style="fill:#0F7986;" width="21.419" height="106.477"/>
<g>
    <path style="fill:#FF5B62;" d="M388.439,252.787l17.277,12.659l-17.881,13.092l-6.49-4.755L388.439,252.787z"/>
    <path style="fill:#FF5B62;" d="M496.828,113.392l9.118,6.681c2.626,1.922,3.198,5.644,1.278,8.267l-37.712,51.479
        c-0.1,0.135-0.29,0.164-0.426,0.066l-18.171-13.312c-0.133-0.1-0.164-0.288-0.066-0.423l37.712-51.479
        c1.922-2.623,5.641-3.198,8.267-1.275L496.828,113.392L496.828,113.392z"/>
</g>
<path style="fill:#ED4C54;" d="M507.224,128.337l-37.723,51.492c-0.1,0.121-0.272,0.149-0.402,0.063l-0.013-0.01l-2.197-1.61
    l40.939-55.884C508.682,124.275,508.526,126.558,507.224,128.337z"/>
<path style="fill:#21D0C3;" d="M382.931,274.945l3.318,2.432l-5.816,7.937c-0.668,0.911-1.962,1.113-2.874,0.444l0,0
    c-0.911-0.668-1.113-1.962-0.444-2.874l5.816-7.937L382.931,274.945L382.931,274.945z"/>
<path style="fill:#FAD24D;" d="M488.36,114.942l22.43,16.432l1.209,0.886l-0.935,1.275l-39.611,54.073l-0.364,0.496l-0.599,0.13
    l-3.909,0.863c-1.69,0.162-2.417-2.513-0.496-3.025l3.31-0.731l38.315-52.302l-21.22-15.545l1.868-2.547v-0.004H488.36z"/>
<rect x="434.783" y="168.459" transform="matrix(0.8067 0.591 -0.591 0.8067 215.3283 -214.8772)" style="fill:#076673;" width="2.656" height="106.477"/>
<g>
    <path style="fill:#15BDB2;" d="M279.015,151.266c-1.162,0-2.106-0.943-2.106-2.106s0.943-2.106,2.106-2.106h34.765l7.802-18.906
        c0.439-1.071,1.666-1.583,2.74-1.145c0.549,0.225,0.951,0.658,1.152,1.17l0.008-0.003l11.572,29.214l9.848-16.677
        c0.591-0.998,1.882-1.33,2.879-0.739c0.316,0.188,0.565,0.444,0.739,0.739l5.712,8.755l8.277-8.803
        c0.794-0.844,2.127-0.886,2.97-0.092c0.159,0.151,0.29,0.319,0.394,0.502l3.357,5.989h45.083c1.162,0,2.106,0.943,2.106,2.106
        c0,1.162-0.943,2.106-2.106,2.106H370v-0.003c-0.737,0-1.45-0.389-1.834-1.074l-2.558-4.564l-8.144,8.658v-0.003
        c-0.115,0.118-0.243,0.23-0.387,0.322c-0.969,0.63-2.273,0.356-2.903-0.617l-5.369-8.235l-10.252,17.361
        c-0.225,0.439-0.604,0.8-1.098,0.993c-1.079,0.426-2.307-0.105-2.733-1.183l-11.24-28.374l-6.294,15.255
        c-0.274,0.844-1.066,1.455-2.001,1.455h-36.168L279.015,151.266z"/>
    <rect x="205.159" y="202.33" style="fill:#15BDB2;" width="74.31" height="8.366"/>
</g>
<g>
    <rect x="206.125" y="225.868" style="fill:#D9DADA;" width="20.816" height="20.816"/>
    <rect x="294.619" y="225.868" style="fill:#D9DADA;" width="20.816" height="20.816"/>
</g>
<rect x="205.159" y="282.895" style="fill:#15BDB2;" width="74.31" height="8.366"/>
<g>
    <rect x="206.125" y="306.443" style="fill:#D9DADA;" width="20.816" height="20.816"/>
    <rect x="294.619" y="306.443" style="fill:#D9DADA;" width="20.816" height="20.816"/>
    <path style="fill:#D9DADA;" d="M274.568,358.717h143.572v4.279H274.568V358.717z M274.568,387.509h103.884v4.279H274.568V387.509z
         M274.568,373.113h143.572v4.279H274.568V373.113z"/>
    <path style="fill:#D9DADA;" d="M233.327,307.416h32.656v4.279h-32.656V307.416z M233.327,320.276h23.63v4.279h-23.63
        L233.327,320.276L233.327,320.276z"/>
    <path style="fill:#D9DADA;" d="M325.148,307.416h32.656v4.279h-32.656V307.416z M325.148,320.276h23.63v4.279h-23.63
        L325.148,320.276L325.148,320.276z"/>
    <path style="fill:#D9DADA;" d="M233.327,226.481h32.656v4.279h-32.656V226.481z M233.327,239.341h23.63v4.279h-23.63
        L233.327,239.341L233.327,239.341z"/>
    <path style="fill:#D9DADA;" d="M325.148,226.481h32.656v4.279h-32.656V226.481z M325.148,239.341h23.63v4.279h-23.63
        L325.148,239.341L325.148,239.341z"/>
</g>
<circle style="fill:#FF5B62;" cx="227.232" cy="145.322" r="28.987"/>
<path style="fill:#FEFEFE;" d="M221.624,124.653h11.227v15.057h15.057v11.227h-15.057v15.057h-11.227v-15.057h-15.057V139.71h15.057
    V124.653z"/>
<g>
    <path style="fill:#0F7986;" d="M296.223,233.805c-0.478-0.486-0.473-1.267,0.016-1.745c0.486-0.478,1.267-0.473,1.745,0.013
        l4.212,4.295l11.407-12.358c0.462-0.502,1.244-0.536,1.748-0.073c0.502,0.462,0.533,1.244,0.073,1.748l-12.288,13.311l0,0
        l-0.042,0.045c-0.486,0.478-1.267,0.473-1.748-0.013l-5.121-5.223L296.223,233.805L296.223,233.805z"/>
    <path style="fill:#0F7986;" d="M212.572,233.805c-0.478-0.486-0.473-1.267,0.016-1.745c0.486-0.478,1.267-0.473,1.745,0.013
        l4.212,4.295l11.407-12.358c0.462-0.502,1.244-0.536,1.748-0.073c0.502,0.462,0.533,1.244,0.073,1.748l-12.288,13.312l0,0
        l-0.042,0.045c-0.486,0.478-1.267,0.473-1.748-0.013l-5.121-5.223h-0.002V233.805z"/>
</g>
<g>
    <path style="fill:#FF5B62;" d="M214.361,308.67c-0.41-0.41-0.41-1.077,0-1.489c0.41-0.41,1.079-0.41,1.489,0l11.836,11.836
        c0.41,0.41,0.41,1.079,0,1.489c-0.41,0.41-1.079,0.41-1.489,0L214.361,308.67z"/>
    <path style="fill:#FF5B62;" d="M227.686,308.67c0.41-0.41,0.41-1.077,0-1.489c-0.41-0.41-1.079-0.41-1.489,0l-11.836,11.836
        c-0.41,0.41-0.41,1.079,0,1.489c0.41,0.41,1.079,0.41,1.489,0L227.686,308.67z"/>
    <path style="fill:#FF5B62;" d="M300.753,307.698c-0.541-0.541-0.541-1.418,0-1.957c0.541-0.541,1.418-0.541,1.957,0l15.566,15.566
        c0.541,0.541,0.541,1.418,0,1.957c-0.541,0.541-1.418,0.541-1.957,0L300.753,307.698z"/>
    <path style="fill:#FF5B62;" d="M318.277,307.698c0.541-0.541,0.541-1.418,0-1.957c-0.541-0.541-1.418-0.541-1.957,0l-15.566,15.566
        c-0.541,0.541-0.541,1.418,0,1.957c0.541,0.541,1.418,0.541,1.957,0L318.277,307.698z"/>
</g>
</svg>
"""

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
    'border': 'none',
    'borderRadius': '6px',
    'cursor': 'pointer',
    'margin-left': '20px',
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
    'paddingLeft': '20px', 
    'paddingRight': '20px',
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
            style={'fontSize': '24px', 'color': '#1e40af', 'cursor': 'pointer'}
        ),
        dbc.Offcanvas(
            [
                dbc.ListGroup(
                    [
                        dbc.ListGroupItem("صفحه اصلی", id="home-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("اهداف داشبورد", id="overview-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("تحلیل‌های اولیه داده‌ها", id="eda-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("تحلیل‌های تکمیلی", id="advanced-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
                        dbc.ListGroupItem("مدل‌های طبقه بندی و ارزیابی", id="models-item", n_clicks=0, style={'cursor': 'pointer', 'fontSize': '18px', 'textAlign': 'right'}),
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

# تعریف layout جدید با هدر
app.layout = html.Div([
    html.Div([
        html.Img(src=f"data:image/svg+xml;utf8,{SVG_CONTENT}", style={'width': '50px', 'height': '50px'}),
        html.H1(id="page-title", style={**HEADER_STYLE, 'margin': '0'}),
        offcanvas
    ], style=MAIN_HEADER_STYLE),
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

# کال‌بک برای تغییر محتوا و عنوان بر اساس انتخاب از off-canvas
@app.callback(
    [Output('page-content', 'children'),
     Output('page-title', 'children')],
    [Input('home-item', 'n_clicks'),
     Input('overview-item', 'n_clicks'),
     Input('eda-item', 'n_clicks'),
     Input('advanced-item', 'n_clicks'),
     Input('models-item', 'n_clicks'),
     Input('predict-item', 'n_clicks'),
     Input('recommendations-item', 'n_clicks')]
)
def update_page(home_clicks, overview_clicks, eda_clicks, advanced_clicks, models_clicks, predict_clicks, recommendations_clicks):
    if not ctx.triggered:
        return [
            html.Div([
                html.H2("نقشه راه سلامت شما", style={'color': '#ffffff', 'textAlign': 'center', 'fontSize': '40px', 'margin-top': '150px'}),
                html.H3("قدرت پیشگیری در دستان شماست", style={'color': '#ffffff', 'textAlign': 'center', 'fontSize': '38px', 'margin': '10px 5px'}),
                html.P("ارزیابی هوشمند احتمال دیابت و دریافت برنامه غذایی و ورزشی کاملا شخصی‌سازی‌شده", style={'color': '#ffffff', 'textAlign': 'center', 'fontSize': '27px', 'margin-bottom': '150px'})
            ], style=BANNER_STYLE),
            "داشبورد تشخیص دیابت"
        ]
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'home-item' and home_clicks:
        return [
            html.Div([
                html.H2("نقشه راه سلامت شما", style={'color': '#ffffff', 'textAlign': 'center', 'fontSize': '36px', 'margin': '10px 0'}),
                html.H3("قدرت پیشگیری در دستان شماست", style={'color': '#ffffff', 'textAlign': 'center', 'fontSize': '24px', 'margin': '10px 0'}),
                html.P("ارزیابی هوشمند احتمال دیابت و دریافت برنامه غذایی و ورزشی کاملا شخصی‌سازی‌شده", style={'color': '#ffffff', 'textAlign': 'center', 'fontSize': '18px', 'margin': '10px 0'})
            ], style=BANNER_STYLE),
            "داشبورد تشخیص دیابت"
        ]

    elif triggered_id == 'overview-item' and overview_clicks:
        return [
            html.Div([
                html.P("""
                اهداف اصلی:
                - تحلیل اکتشافی داده‌ها (EDA) برای درک توزیع و روابط ویژگی‌ها.
                - تشخیص و مدیریت ناهنجاری‌ها و مقادیر گمشده.
                - آموزش مدل‌های مختلف کلاسیفیکیشن و انتخاب بهترین (Gradient Boosting بعد از حذف ناهنجاری‌ها).
                - ایجاد داشبورد برای پیش‌بینی دیابت و پیشنهاد برنامه‌های شخصی‌سازی‌شده بر اساس ویژگی‌های کلیدی.
                این داشبورد با Dash ساخته شده و روی Render دیپلوی می‌شود.
                """, style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}),
                html.Img(src='/img/img1.png', style={'width': '50%', 'margin': '20px auto', 'display': 'block'}, alt="تصویر داشبورد"),
                html.P("تصویر بارگذاری نشد.", style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '16px', 'color': '#FF0000', 'margin': '10px'}),
                html.P("""
                مزایا و کاربردهای داشبورد:
                این داشبورد امکان پیش‌بینی دقیق احتمال ابتلا به دیابت را فراهم می‌کند و پیشنهادات شخصی‌سازی‌شده برای رژیم غذایی و ورزش ارائه می‌دهد. مزایای آن شامل دقت بالا در مدل‌سازی، دسترسی آسان برای کاربران، و کمک به پیشگیری از بیماری است. کاربردها عبارتند از استفاده در کلینیک‌ها برای غربالگری، ادغام با اپلیکیشن‌های سلامت، و تحقیقات پزشکی برای تحلیل داده‌های بزرگ.
                """, style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'})
            ]),
            "اهداف داشبورد"
        ]

    elif triggered_id == 'eda-item' and eda_clicks:
        figs = []
        for col in df.columns:
            fig_hist = px.histogram(df, x=col, nbins=20, title=f"هیستوگرام {col}")
            figs.append(dcc.Graph(figure=fig_hist, style=GRAPH_STYLE))
        fig_box = px.box(df, y=['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure'], title="باکس پلات ویژگی‌ها")
        fig_scatter = px.scatter(df, x='Glucose', y='BMI', color='Outcome', title='اسکتر پلات Glucose vs BMI', color_continuous_scale='RdBu')
        return [
            html.Div([
                *figs,
                dcc.Graph(figure=fig_box, style=GRAPH_STYLE),
                dcc.Graph(figure=fig_scatter, style=GRAPH_STYLE)
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            "تحلیل‌های اولیه داده‌ها"
        ]

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
            [
                dcc.Graph(figure=fig_corr, style=GRAPH_STYLE),
                html.P(f"تعداد داده‌های حذف شده: {num_removed}", style={'direction': 'rtl', 'text-align': 'right', 'fontSize': '18px', 'margin': '20px'}),
                html.P("""
                تحلیل تکمیلی شامل:
                - شناسایی ناهنجاری‌ها با IsolationForest.
                - رگرسیون خطی برای پیش‌بینی Glucose بر اساس BMI.
                - ماتریس کورلیشن برای بررسی روابط.
                """, style={**BASE_STYLE, 'margin': '20px'})
            ],
            "تحلیل‌های تکمیلی"
        ]

    elif triggered_id == 'models-item' and models_clicks:
        results_df = pd.DataFrame({'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'Gradient Boosting', 'LightGBM', 'MLP'],
                                 'Accuracy': [0.75, '0.72', 0.70, 0.78, 0.80, 0.82, 0.79, 0.76]})
        fig_cm = px.imshow([[50, 10], [8, 60]], text_auto=True, color_continuous_scale='Blues', title='ماتریس درهم‌ریختگی (نمونه)')
        return [
            [
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
            ],
            "مدل‌های طبقه‌بندی و ارزیابی"
        ]

    elif triggered_id == 'predict-item' and predict_clicks:
        return [
            html.Div([
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
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            "پیش‌بینی دیابت"
        ]

    elif triggered_id == 'recommendations-item' and recommendations_clicks:
        return [
            html.Div([
                html.Label("بر اساس ویژگی‌های کلیدی: Glucose, BMI, Age, Insulin, BloodPressure, Pregnancies", style={**BASE_STYLE, 'margin': '20px'}),
                *[html.Div([
                    html.Label(feature, style={**BASE_STYLE, 'margin': '20px'}),
                    dcc.Input(id=f'rec-input-{feature}', type='number', value=0 if feature in ['Age', 'Pregnancies'] else 0.0, step=1 if feature in ['Age', 'Pregnancies'] else 0.1, style=INPUT_STYLE)
                ]) for feature in ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure', 'Pregnancies']],
                html.Button('دریافت پیشنهاد', id='recommend-button', n_clicks=0, style=BUTTON_STYLE),
                html.Button('دانلود برنامه به صورت PDF', id='download-button', n_clicks=0, style={**BUTTON_STYLE, 'backgroundColor': '#008000', 'margin': '20px'}),
                dcc.Download(id='download-pdf'),
                html.Div(id='recommend-output', style=OUTPUT_STYLE)
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            "پیشنهاد برنامه غذایی و ورزشی"
        ]

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
