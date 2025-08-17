import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer

# تنظیمات صفحه
st.set_page_config(page_title="تشخیص دیابت", layout="wide")

# CSS برای آیکون مشاوره آنلاین
st.markdown("""
<style>
.chat-icon {
    position: fixed;
    bottom: 20px;
    left: 20px;
    background-color: #007bff;
    color: white;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    font-size: 24px;
}
.chat-icon:hover {
    background-color: #0056b3;
}
</style>
""", unsafe_allow_html=True)

# آیکون مشاوره آنلاین
st.markdown("""
<div class='chat-icon' onclick='window.open("https://t.me/your_chat_link", "_blank")'>💬</div>
""", unsafe_allow_html=True)

# معرفی پروژه
st.title("پروژه داده‌کاوی: تشخیص دیابت")
st.markdown("""
### معرفی پروژه
این پروژه با استفاده از دیتاست **Pima Indians Diabetes** انجام شده است. هدف، پیش‌بینی احتمال ابتلا به دیابت با استفاده از ویژگی‌هایی مانند گلوکز، فشار خون، BMI و ... است.
### اهداف
- تحلیل داده‌های دیتاست (توزیع ویژگی‌ها، همبستگی)
- پیش‌بینی با مدل Random Forest (دقت ~94%)
- پیشنهاد رژیم غذایی تخصصی بر اساس ویژگی‌های ورودی
- امکان مشاوره آنلاین با متخصص
""")

# لود دیتاست
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df[cols_to_replace] = imputer.fit_transform(df[cols_to_replace])
    return df

df = load_data()

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore

# پیش‌پردازش و اضافه کردن ناهنجاری‌ها (مشابه diabetest.py)
features = ['BMI', 'Glucose', 'Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
X_raw = df[features]
scaler = joblib.load("scaler.pkl")  # اسکیلر رو اینجا لود نکن، بعداً استفاده می‌کنیم
X_scaled = scaler.fit_transform(X_raw.drop('Outcome', axis=1))  # فیت و transform جدید
iso_model = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly_ISO'] = (iso_model.fit_predict(X_scaled) == -1).astype(int)
# سایر ناهنجاری‌ها (KNN, Z-Score, IQR) رو هم می‌تونی اضافه کنی، ولی حداقل Anomaly_ISO کافیه
# جایگزینی مقادیر صفر با میانه
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[cols_to_replace] = imputer.fit_transform(df[cols_to_replace])

# تحلیل داده‌ها
st.header("تحلیل داده‌ها")
st.subheader("آمار توصیفی")
st.dataframe(df.describe())
st.caption("جدول بالا آمار توصیفی ویژگی‌های دیتاست را نشان می‌دهد.")

# هیستوگرام ویژگی‌ها
st.subheader("هیستوگرام ویژگی‌ها")
feature = st.selectbox("ویژگی را انتخاب کنید:", df.columns[:-1])  # بدون ستون Outcome
fig = px.histogram(df, x=feature, color="Outcome", title=f"توزیع {feature}",
                   labels={"Outcome": "وضعیت دیابت (0=غیر دیابتی، 1=دیابتی)"})
st.plotly_chart(fig)
st.caption(f"این نمودار توزیع {feature} را برای بیماران دیابتی و غیر دیابتی نشان می‌دهد.")

# باکس‌پلات
st.subheader("باکس‌پلات ویژگی‌ها")
features_to_plot = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df[features_to_plot], ax=ax)
plt.title("باکس‌پلات ویژگی‌های منتخب")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)
st.caption("این باکس‌پلات توزیع و ناهنجاری‌های احتمالی در ویژگی‌های منتخب را نشان می‌دهد.")

# اسکترپلات
st.subheader("اسکترپلات گلوکز در مقابل BMI")
fig = px.scatter(df, x='Glucose', y='BMI', color='Outcome', title='گلوکز در مقابل BMI',
                 labels={"Outcome": "وضعیت دیابت (0=غیر دیابتی، 1=دیابتی)"})
st.plotly_chart(fig)
st.caption("این نمودار رابطه بین گلوکز و BMI را با رنگ‌بندی بر اساس وضعیت دیابت نشان می‌دهد.")

# ماتریس کورلیشن
st.subheader("ماتریس کورلیشن")
corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
st.caption("این ماتریس همبستگی بین ویژگی‌ها را نشان می‌دهد. مقادیر نزدیک به 1 یا -1 نشان‌دهنده همبستگی قوی است.")

# نتایج مدل
st.header("نتایج مدل")
st.markdown("""
مدل **Random Forest** با دقت **~94%** بهترین عملکرد را قبل از حذف ناهنجاری‌ها داشته است.
""")

try:
    model_rf = joblib.load("random_forest_model.pkl")
    scaler = StandardScaler()  # بازسازی اسکیلر
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_scaled = scaler.fit_transform(X)  # فیت با دیتاست فعلی
    y_pred_rf = model_rf.predict(X_scaled)
    results_df = pd.DataFrame([
        {'Model': 'Random Forest', 'Accuracy': accuracy_score(y, y_pred_rf), 
         'Precision': precision_score(y, y_pred_rf), 'Recall': recall_score(y, y_pred_rf), 
         'F1-Score': f1_score(y, y_pred_rf), 'ROC-AUC': roc_auc_score(y, y_pred_rf)}
    ])
    st.dataframe(results_df)
    st.caption("این جدول معیارهای عملکرد مدل Random Forest را نشان می‌دهد.")

    # ماتریس درهم‌ریختگی
    st.subheader("ماتریس درهم‌ریختگی (Random Forest)")
    cm = confusion_matrix(y, y_pred_rf)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title("ماتریس درهم‌ریختگی - Random Forest")
    plt.ylabel("واقعی")
    plt.xlabel("پیش‌بینی‌شده")
    st.pyplot(fig)
    st.caption("این ماتریس پیش‌بینی‌های درست و نادرست مدل Random Forest را نشان می‌دهد.")
except Exception as e:
    st.error(f"خطا در لود مدل یا محاسبه ماتریس: {e}")
    
# پیش‌بینی و پیشنهاد رژیم غذایی
st.header("پیش‌بینی و پیشنهاد رژیم غذایی")
st.write("مقادیر ویژگی‌ها را وارد کنید:")
with st.form("predict_form"):
    pregnancies = st.number_input("تعداد بارداری‌ها", min_value=0, value=0)
    glucose = st.number_input("گلوکز", min_value=0.0, value=100.0)
    blood_pressure = st.number_input("فشار خون", min_value=0.0, value=70.0)
    skin_thickness = st.number_input("ضخامت پوست", min_value=0.0, value=20.0)
    insulin = st.number_input("انسولین", min_value=0.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    dpf = st.number_input("تابع شجره دیابت", min_value=0.0, value=0.5)
    age = st.number_input("سن", min_value=0, value=30)

    submitted = st.form_submit_button("پیش‌بینی و دریافت رژیم")
    if submitted:
        # آماده‌سازی داده ورودی
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]])
        input_data_scaled = scaler.transform(input_data)
        # پیش‌بینی با Random Forest
        prediction = model_rf.predict(input_data_scaled)
        result = 'دیابت' if prediction[0] == 1 else 'غیر دیابتی'
        st.write(f"**نتیجه پیش‌بینی (Random Forest)**: {result}")

        # پیشنهاد رژیم غذایی تخصصی
        st.subheader("پیشنهاد رژیم غذایی تخصصی")
        diet_suggestions = []

        if prediction[0] == 1:  # دیابتی
            st.markdown("**رژیم پیشنهادی برای بیماران دیابتی (بر اساس دستورالعمل ADA 2025):**")
            # بر اساس Glucose
            if glucose > 126:
                diet_suggestions.append("""
                - **برای گلوکز بالا**: رژیم کم‌کربوهیدرات (26-45% کالری از کربوهیدرات) یا خیلی کم‌کربوهیدرات (<26%، حدود 20-50 گرم کربوهیدرات غیرفیبری در روز). تمرکز روی سبزیجات غیرنشاسته‌ای (مثل بروکلی، اسفناج)، چربی‌های سالم (آووکادو، روغن زیتون) و پروتئین کم‌چربی (مرغ، ماهی) برای کنترل قند خون.
                """)
            else:
                diet_suggestions.append("""
                - **برای گلوکز کنترل‌شده**: رژیم مدیترانه‌ای با تمرکز روی غذاهای گیاهی، ماهی (مثل سالمون)، روغن زیتون و مصرف محدود لبنیات و تخم‌مرغ برای حفظ کنترل قند.
                """)

            # بر اساس BMI
            if bmi > 30:
                diet_suggestions.append("""
                - **برای BMI بالا (چاقی)**: رژیم خیلی کم‌چربی (70-77% کربوهیدرات با 30-60 گرم فیبر، <10% چربی) یا کم‌کربوهیدرات برای کاهش وزن (هدف: کاهش 3-7% وزن بدن، تا 15% برای remission احتمالی). شامل سبزیجات، حبوبات و پروتئین‌های گیاهی.
                """)
            elif bmi > 25:
                diet_suggestions.append("""
                - **برای BMI متوسط (اضافه وزن)**: رژیم گیاه‌خواری یا وگان با تمرکز روی غذاهای گیاهی، سبزیجات (مثل کدو، کلم) و میوه‌های کم‌قند (مثل توت‌ها) برای مدیریت وزن و کاهش A1C.
                """)
            else:
                diet_suggestions.append("""
                - **برای BMI نرمال**: رژیم متعادل DASH با تمرکز روی میوه‌ها (سیب، گلابی)، سبزیجات، غلات کامل (جو، کینوا) و پروتئین کم‌چربی برای حفظ وزن.
                """)

            # بر اساس Age
            if age > 65:
                diet_suggestions.append("""
                - **برای افراد مسن**: رژیم مدیترانه‌ای یا DASH برای سلامت قلب، با تمرکز روی پروتئین (مثل تخم‌مرغ، ماهی) برای حفظ عضله، وعده‌های کوچک منظم و اجتناب از هیپوگلیسمی. شامل غذاهای مغذی مثل مغزها و سبزیجات.
                """)
            elif age > 50:
                diet_suggestions.append("""
                - **برای افراد میانسال**: رژیم کم‌چربی (<30% کالری از چربی، <10% چربی اشباع) با تمرکز روی سبزیجات، میوه‌ها، کربوهیدرات‌های پیچیده (مثل نان سبوس‌دار) و پروتئین کم‌چربی برای جلوگیری از عوارض.
                """)
            else:
                diet_suggestions.append("""
                - **برای افراد جوان**: رژیم کم‌کربوهیدرات با تمرکز روی سبزیجات، چربی‌های سالم (مثل مغزها) و کنترل بخش برای فعالیت روزانه.
                """)

            # بر اساس BloodPressure
            if blood_pressure > 130:
                diet_suggestions.append("""
                - **برای فشار خون بالا**: رژیم DASH با تمرکز روی سبزیجات، میوه‌ها، لبنیات کم‌چربی، غلات کامل، مرغ، ماهی و مغزها. کاهش نمک (کمتر از 1500 میلی‌گرم در روز)، گوشت قرمز و شیرینی‌ها برای کنترل فشار خون.
                """)
            else:
                diet_suggestions.append("""
                - **برای فشار خون نرمال**: رژیم خیلی کم‌چربی یا کم‌کربوهیدرات برای حفظ فشار خون و کاهش تری‌گلیسرید. شامل غذاهایی مثل آووکادو و ماهی.
                """)

        else:  # غیر دیابتی
            st.markdown("**رژیم پیشنهادی پیشگیرانه برای افراد غیر دیابتی (بر اساس دستورالعمل ADA 2025):**")
            # بر اساس Glucose
            if glucose > 100:
                diet_suggestions.append("""
                - **برای گلوکز نسبتاً بالا**: رژیم مدیترانه‌ای برای کاهش ریسک دیابت، با غذاهای گیاهی (مثل حبوبات)، ماهی و روغن زیتون.
                """)
            else:
                diet_suggestions.append("""
                - **برای گلوکز نرمال**: رژیم متعادل با تمرکز روی فیبر بالا (مثل سبزیجات، غلات کامل) و کنترل بخش.
                """)

            # بر اساس BMI
            if bmi > 30:
                diet_suggestions.append("""
                - **برای BMI بالا**: رژیم کم‌چربی یا گیاه‌خواری برای کاهش وزن و جلوگیری از دیابت. شامل سبزیجات، میوه‌های کم‌قند و پروتئین‌های گیاهی.
                """)
            elif bmi > 25:
                diet_suggestions.append("""
                - **برای BMI متوسط**: رژیم DASH برای مدیریت وزن و سلامت قلب، با تمرکز روی سبزیجات، میوه‌ها و غلات کامل.
                """)
            else:
                diet_suggestions.append("""
                - **برای BMI نرمال**: رژیم وگان یا مدیترانه‌ای برای حفظ سلامت، شامل میوه‌ها، سبزیجات و چربی‌های سالم.
                """)

            # بر اساس Age
            if age > 65:
                diet_suggestions.append("""
                - **برای افراد مسن**: رژیم DASH برای سلامت قلب و فشار خون، با تمرکز روی غذاهای مغذی مثل ماهی و مغزها.
                """)
            elif age > 50:
                diet_suggestions.append("""
                - **برای افراد میانسال**: رژیم کم‌کربوهیدرات برای جلوگیری از افزایش وزن، با تمرکز روی سبزیجات و پروتئین کم‌چربی.
                """)
            else:
                diet_suggestions.append("""
                - **برای افراد جوان**: رژیم متعادل با فعالیت بدنی، شامل غلات کامل، میوه‌ها و سبزیجات.
                """)

            # بر اساس BloodPressure
            if blood_pressure > 130:
                diet_suggestions.append("""
                - **برای فشار خون بالا**: رژیم DASH برای کاهش فشار خون و ریسک دیابت، با کاهش نمک و تمرکز روی سبزیجات و میوه‌ها.
                """)
            else:
                diet_suggestions.append("""
                - **برای فشار خون نرمال**: رژیم مدیترانه‌ای برای پیشگیری کلی، با غذاهای گیاهی و چربی‌های سالم.
                """)

        # نمایش پیشنهادات
        for suggestion in diet_suggestions:
            st.markdown(suggestion)

        st.markdown("""
        **نکته**: این پیشنهادات کلی و بر اساس دستورالعمل‌های ADA هستند. برای رژیم شخصی‌سازی‌شده، با متخصص تغذیه مشورت کنید.
        """)
