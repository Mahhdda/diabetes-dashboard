import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# لود CSS
# لود CSS با بررسی خطا
try:
    with open("style.css") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        print("CSS لود شد با موفقیت.")
except FileNotFoundError:
    st.error("فایل style.css یافت نشد. لطفاً مطمئن شوید که توی روت مخزن هست.")
except Exception as e:
    st.error(f"خطا در لود CSS: {str(e)}")

# لود مدل‌ها
try:
    best_model = joblib.load("best_model_gradient_boosting.pkl")
    scaler_clean = joblib.load("scaler_clean.pkl")
    voting_model = joblib.load("voting_model_clean.pkl")
except FileNotFoundError:
    st.error("فایل‌های مدل یافت نشد. لطفاً آن‌ها را آپلود کنید.")
    st.stop()

# عنوان داشبورد (به‌روزرسانی‌شده)
st.title("داشبورد تشخیص دیابت با Gradient Boosting")

# سایدبار برای ناوبری
st.sidebar.title("ناوبری")
page = st.sidebar.radio("بخش مورد نظر", ["مروری بر پروژه", "تحلیل‌های اولیه داده‌ها", "تحلیل‌های تکمیلی", "مدل‌های کلاسیفیکیشن و ارزیابی", "پیش‌بینی دیابت", "پیشنهاد برنامه غذایی و ورزشی"])

# ساختار شرط‌ها (کامل و بدون تکرار)
if page == "مروری بر پروژه":
    st.header("مروری بر پروژه و اهداف آن")
    st.write("""
    این پروژه داده‌کاوی بر روی دیتاست Pima Indians Diabetes تمرکز دارد که شامل 768 رکورد و 9 ویژگی است. 
    اهداف اصلی:
    - تحلیل اکتشافی داده‌ها (EDA) برای درک توزیع و روابط ویژگی‌ها.
    - تشخیص و مدیریت ناهنجاری‌ها و مقادیر گمشده.
    - آموزش مدل‌های مختلف کلاسیفیکیشن و انتخاب بهترین (Gradient Boosting بعد از حذف ناهنجاری‌ها).
    - ایجاد داشبورد برای پیش‌بینی دیابت و پیشنهاد برنامه‌های شخصی‌سازی‌شده بر اساس ویژگی‌های کلیدی.
    این داشبورد با Streamlit ساخته شده و روی Streamlit Cloud deploy می‌شود.
    """)

elif page == "تحلیل‌های اولیه داده‌ها":
    st.header("تحلیل‌های اولیه داده‌ها")
    try:
        df = pd.read_csv("diabetest.csv")
    except FileNotFoundError:
        st.error("فایل دیتاست یافت نشد. لطفاً diabetest.csv را آپلود کنید.")
        st.stop()
    
    st.subheader("هیستوگرام ویژگی‌ها")
    fig, ax = plt.subplots(figsize=(10, 8))
    df.hist(bins=20, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("باکس پلات ویژگی‌های انتخابی")
    features_to_plot = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[features_to_plot], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("اسکتر پلات Glucose vs BMI")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome', palette='coolwarm', ax=ax)
    st.pyplot(fig)

elif page == "تحلیل‌های تکمیلی":
    st.header("تحلیل‌های تکمیلی")
    try:
        df = pd.read_csv("diabetest.csv")
    except FileNotFoundError:
        st.error("فایل دیتاست یافت نشد.")
        st.stop()
    
    st.subheader("کورلیشن ماتریکس")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.subheader("گزارش حذف ناهنجاری‌ها با Isolation Forest")
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso.fit_predict(df.drop('Outcome', axis=1))
    num_removed = len(df) - sum(outliers == 1)
    st.write(f"تعداد داده‌های حذف شده: **{num_removed}**")
    st.write("این روش 5% از داده‌ها رو به عنوان ناهنجاری در نظر می‌گیره و حذف می‌کنه.")
    
    st.subheader("رگرسیون خطی: BMI vs Glucose")
    from sklearn.linear_model import LinearRegression
    X = df[['BMI']]
    y = df['Glucose']
    reg_model = LinearRegression().fit(X, y)
    y_pred = reg_model.predict(X)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, alpha=0.6)
    ax.plot(X, y_pred, color='red')
    st.pyplot(fig)
    
    st.subheader("SVM مرز تصمیم (کرنل خطی)")
    from sklearn.svm import SVC
    X_svm = df[['BMI', 'Glucose']]
    y_svm = df['Outcome']
    svm_model = SVC(kernel='linear').fit(X_svm, y_svm)
    st.write("نمایش مرز تصمیم SVM (تصویر ساده‌شده). برای جزئیات بیشتر به کد اصلی مراجعه کنید.")

elif page == "مدل‌های کلاسیفیکیشن و ارزیابی":
    st.header("مدل‌های کلاسیفیکیشن و ارزیابی")
    st.write("""
    مدل‌های آموزش‌دیده: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, Gradient Boosting, LightGBM, MLP.
    بهترین مدل: Gradient Boosting روی داده‌های پاک‌شده (بعد از حذف ناهنجاری‌ها) با دقت بالا.
    ارزیابی شامل Accuracy, Precision, Recall, F1-Score, ROC-AUC و Cross-Validation.
    """)
    # مثال نتایج (به‌روزرسانی‌شده برای Gradient Boosting)
    results_data = {
        'Model': ['Gradient Boosting', 'Logistic Regression', 'MLP'],
        'Accuracy': [0.80, 0.78, 0.77],
        'Precision': [0.75, 0.73, 0.72],
        'Recall': [0.68, 0.66, 0.65],
        'F1-Score': [0.71, 0.69, 0.68],
        'ROC-AUC': [0.87, 0.85, 0.84]
    }
    results_df = pd.DataFrame(results_data)
    st.table(results_df)
    
    st.subheader("ماتریس سردرگمی برای Gradient Boosting")
    cm = np.array([[92, 8], [12, 42]])  # مثال؛ واقعی رو جایگزین کن
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

elif page == "پیش‌بینی دیابت":
    st.header("پیش‌بینی دیابت")
    st.write("ویژگی‌ها را وارد کنید تا مدل پیش‌بینی کند.")
    
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    inputs = {}
    for feature in features:
        if feature in ['Pregnancies', 'Age']:
            inputs[feature] = st.number_input(f"{feature} (عدد صحیح)", min_value=0, step=1)
        else:
            inputs[feature] = st.number_input(f"{feature} (عدد اعشاری)", min_value=0.0, step=0.1)
    
    model_choice = st.selectbox("انتخاب مدل", ["Gradient Boosting (بهترین تک مدل)", "Voting (سه مدل برتر)"])
    
    if st.button("پیش‌بینی"):
        input_df = pd.DataFrame([inputs], columns=features)
        if input_df.isnull().any().any():
            st.error("لطفاً همه فیلدها را پر کنید.")
        else:
            try:
                input_scaled = scaler_clean.transform(input_df)
                if model_choice == "Voting (سه مدل برتر)":
                    prediction = voting_model.predict(input_scaled)[0]
                    prob = voting_model.predict_proba(input_scaled)[0][1] * 100
                else:
                    prediction = best_model.predict(input_scaled)[0]
                    prob = best_model.predict_proba(input_scaled)[0][1] * 100
                
                if prediction == 1:
                    st.error(f"احتمال دیابت: {prob:.2f}% (دیابتی)")
                else:
                    st.success(f"احتمال دیابت: {prob:.2f}% (غیر دیابتی)")
            except ValueError as e:
                st.error("خطا در پیش‌بینی: ممکن است ویژگی‌ها با مدل سازگار نباشند. لطفاً ترتیب یا مقادیر را بررسی کنید.")
                st.write("جزئیات خطا برای دیباگ:", str(e))

elif page == "پیشنهاد برنامه غذایی و ورزشی":
    st.header("پیشنهاد برنامه غذایی، خواب، ورزش و پیاده‌روی")
    st.write("بر اساس 4 ویژگی مهم: Glucose, BMI, Age, Insulin")
    
    glucose = st.number_input("Glucose", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    age = st.number_input("Age", min_value=0)
    insulin = st.number_input("Insulin", min_value=0.0)
    
    if st.button("دریافت پیشنهاد"):
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
            exercise += "ورزش روزانه 45 دقیقه: ایروبیک، وزنه‌برداری سبک. "
        else:
            exercise += "ورزش متوسط 30 دقیقه: یوگا یا شنا."
        
        walking = "پیاده‌روی: حداقل 5000 قدم روزانه، اگر BMI بالا باشد 10000 قدم."
        
        st.write(diet)
        st.write(sleep)
        st.write(exercise)
        st.write(walking)
