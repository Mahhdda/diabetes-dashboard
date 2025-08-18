import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib

# تنظیمات صفحه
st.set_page_config(page_title="پروژه تشخیص دیابت", layout="wide")

# CSS برای آیکون مشاوره آنلاین (سمت راست پایین)
st.markdown("""
<style>
.chat-icon {
    position: fixed;
    bottom: 20px;
    right: 20px;  /* سمت راست پایین */
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
<div class='chat-icon' onclick='window.open("https://t.me/your_chat_link", "_blank")'>💬</div>  <!-- لینک به چت مشاور -->
""", unsafe_allow_html=True)

# لود دیتاست با fallback
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        st.warning("دیتاست پیدا نشد! در حال لود از URL عمومی...")
        try:
            url = "https://raw.githubusercontent.com/Mahhdda/diabetes-dashboard/main/diabetes.csv"
            df = pd.read_csv(url)
        except Exception as e:
            st.error(f"لود دیتاست ناموفق بود! خطا: {e}")
            st.stop()
    return df

df = load_data()

# پیش‌پردازش دیتاست (مشابه diabetest.py)
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[cols_to_replace] = imputer.fit_transform(df[cols_to_replace])

# معرفی پروژه و اهداف
st.title("پروژه داده‌کاوی: تشخیص دیابت")
st.markdown("""
### معرفی پروژه
این پروژه با استفاده از دیتاست **Pima Indians Diabetes** انجام شده است. این دیتاست شامل اطلاعات پزشکی 768 زن از قبیله پیما است که شامل ویژگی‌هایی مانند تعداد بارداری‌ها، گلوکز، فشار خون، ضخامت پوست، انسولین، BMI، تابع شجره دیابت، سن و نتیجه (Outcome: 0=غیر دیابتی، 1=دیابتی) است.
### اهداف
- تحلیل اکتشافی داده‌ها (EDA) برای درک توزیع و روابط ویژگی‌ها.
- تشخیص ناهنجاری‌ها با روش‌های مختلف مانند Isolation Forest، KNN، Z-Score و IQR.
- آموزش مدل‌های یادگیری ماشین برای پیش‌بینی دیابت.
- نمایش نتایج مدل‌ها به صورت ایستا و تعاملی.
- پیشنهاد رژیم غذایی، ساعت خواب مناسب و ورزش‌های اصلاحی بر اساس ورودی کاربر.
- ارائه توضیحات متنی واضح برای کاربران.
این داشبورد با Streamlit ساخته شده و می‌تواند برای دمو یا ویدیو عملکرد سایت استفاده شود.
""")

# تحلیل داده‌ها (EDA)
st.header("تحلیل داده‌ها")
st.subheader("آمار توصیفی")
st.dataframe(df.describe())
st.caption("این جدول آمار توصیفی ویژگی‌های دیتاست را نشان می‌دهد، شامل میانگین، میانه، انحراف استاندارد و غیره.")

st.subheader("هیستوگرام ویژگی‌ها")
feature = st.selectbox("ویژگی را انتخاب کنید:", df.columns[:-1])
fig, ax = plt.subplots()
df[feature].hist(bins=20, ax=ax)
plt.title(f"هیستوگرام {feature}")
st.pyplot(fig)
st.caption(f"این نمودار توزیع {feature} را نشان می‌دهد.")

st.subheader("باکس‌پلات ویژگی‌های منتخب")
features_to_plot = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df[features_to_plot], ax=ax)
plt.title("باکس‌پلات ویژگی‌های منتخب")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)
st.caption("این باکس‌پلات توزیع و ناهنجاری‌های احتمالی در ویژگی‌های منتخب را نشان می‌دهد.")

st.subheader("اسکترپلات گلوکز در مقابل BMI")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome', palette='coolwarm', ax=ax)
plt.title('اسکترپلات گلوکز در مقابل BMI')
st.pyplot(fig)
st.caption("این نمودار رابطه بین گلوکز و BMI را با رنگ‌بندی بر اساس وضعیت دیابت نشان می‌دهد.")

st.subheader("ماتریس کورلیشن")
corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
st.caption("این ماتریس همبستگی بین ویژگی‌ها را نشان می‌دهد. مقادیر نزدیک به 1 یا -1 نشان‌دهنده همبستگی قوی است.")

# مدل‌های یادگیری ماشین (مشابه diabetest.py)
st.header("مدل‌های یادگیری ماشین")
st.markdown("مدل‌ها روی داده پیش‌پردازش‌شده آموزش دیده‌اند. نتایج شامل accuracy، precision، recall، F1-score و ROC-AUC است.")

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }
    results.append(metrics)

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
st.subheader("نتایج مدل‌ها")
st.dataframe(results_df)
st.caption("این جدول نتایج مدل‌ها را بر اساس معیارهای مختلف نشان می‌دهد. مدل Random Forest معمولاً بهترین عملکرد را دارد.")

# نمایش confusion matrix برای مدل انتخابی
selected_model = st.selectbox("مدل را برای confusion matrix انتخاب کنید:", list(models.keys()))
model = models[selected_model]
y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title(f"Confusion Matrix - {selected_model}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
st.pyplot(fig)
st.caption("این ماتریس پیش‌بینی‌های درست و نادرست مدل را نشان می‌دهد.")

# بخش پیشنهاد رژیم غذایی, خواب, و ورزش
st.header("پیشنهاد رژیم غذایی, خواب و ورزش")
st.markdown("بر اساس 4 ویژگی (Glucose, BMI, Age, BloodPressure) که وارد می‌کنید، پیشنهادهای شخصی‌سازی‌شده ارائه می‌شود.")

with st.form("suggestion_form"):
    glucose = st.number_input("گلوکز", min_value=0.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    age = st.number_input("سن", min_value=0, value=30)
    blood_pressure = st.number_input("فشار خون", min_value=0.0, value=70.0)
    submitted = st.form_submit_button("دریافت پیشنهاد")

    if submitted:
        # منطق پیشنهاد (بر اساس قوانین ساده، الهام‌گرفته از ADA)
        is_diabetic = (glucose > 126 or bmi > 30 or blood_pressure > 130)  # شرط ساده برای تشخیص ریسک دیابت

        st.subheader("پیشنهادها")
        if is_diabetic:
            st.markdown("**ریسک دیابت بالا تشخیص داده شد. پیشنهادهای زیر برای کنترل دیابت:**")
            # رژیم غذایی
            diet = "رژیم کم‌کربوهیدرات (کمتر از 50 گرم کربوهیدرات روزانه)، تمرکز روی سبزیجات غیرنشاسته‌ای (بروکلی, اسفناج), پروتئین کم‌چربی (مرغ, ماهی), و چربی‌های سالم (آووکادو, روغن زیتون). اجتناب از قند و غذاهای فرآوری‌شده."
            # خواب
            sleep = "8-9 ساعت خواب شبانه, با برنامه منظم (خواب قبل از 11 شب و بیداری قبل از 7 صبح) برای کنترل قند خون."
            # ورزش
            exercise = "ورزش‌های اصلاحی: پیاده‌روی سریع 30 دقیقه روزانه, یوگا برای کاهش استرس, و تمرینات قدرتی (وزنه‌برداری سبک) 3 بار در هفته."
        else:
            st.markdown("**ریسک دیابت پایین تشخیص داده شد. پیشنهادهای زیر برای پیشگیری:**")
            diet = "رژیم متعادل مدیترانه‌ای با میوه‌ها (سیب, توت‌ها), سبزیجات, غلات کامل (جو, کینوا), و پروتئین گیاهی. مصرف قند محدود."
            sleep = "7-8 ساعت خواب شبانه, با تمرکز روی کیفیت خواب (اجتناب از صفحه نمایش قبل از خواب)."
            exercise = "ورزش‌های اصلاحی: دویدن یا شنا 45 دقیقه روزانه, پیلاتس برای تقویت هسته بدن, و دوچرخه‌سواری برای سلامت قلب."

        st.markdown(f"**رژیم غذایی پیشنهادی:** {diet}")
        st.markdown(f"**ساعت خواب مناسب:** {sleep}")
        st.markdown(f"**ورزش‌های اصلاحی و مناسب:** {exercise}")
        st.caption("این پیشنهادها کلی هستند و بر اساس دستورالعمل ADA. برای مشاوره شخصی، با متخصص تماس بگیرید.")

# توضیحات اضافی
st.header("دمو و ویدیو عملکرد")
st.markdown("برای دمو, از این داشبورد استفاده کنید. می‌توانید یک ویدیو کوتاه از عملکرد سایت (با ابزارهایی مانند Loom) ضبط کنید و به استاد ارائه دهید. لینک ویدیو را اینجا قرار دهید: [لینک ویدیو دمو](https://your-demo-link.com)")
st.caption("این بخش برای پوشش نیاز ارائه دمو یا ویدیو است.")

# ذخیره مدل برای استفاده آینده (اختیاری)
try:
    joblib.dump(models["Random Forest"], "random_forest_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
except Exception as e:
    st.warning(f"ذخیره مدل ناموفق بود: {e}")
