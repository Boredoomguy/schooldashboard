import pandas as pd
import pdfplumber
import numpy as np
import streamlit as st

features = ["attendance", "homework", "quiz", "arabic", "math", "english"]

features_ar = {
    "attendance": "حضور",
    "homework": "واجب",
    "quiz": "اختبار",
    "arabic": "العربية",
    "math": "الرياضيات",
    "english": "الإنجليزية"
}
class SimpleModel:

    def __init__(self): 
        self.weights = None
        self.bias = 0
        self.trained = False
    def normalize(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
    def train(self, x, y, training_num=300, lr=0.3):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        self.x_min = x.min(axis=0)
        self.x_max = x.max(axis=0)
        x = self.normalize(x)
        self.weights = np.zeros(x.shape[1])
        self.bias = 0.0
        self.loss_history = []
        for epoch in range(training_num):
            pred = x @ self.weights + self.bias
            error = pred - y
            self.loss_history.append(np.mean(error ** 2))
            self.weights -= lr * (x.T @ error) / len(x)
            self.bias -= lr * error.mean()
        self.trained = True
    def predict_all(self,x): # makes the prediction for all the data
        x = np.array(x, dtype=float)
        x = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
        return np.clip(x @ self.weights + self.bias, 0, 100)
def extract_table_from_pdf(uploaded_file):
    all_rows = []
    header = None
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            if header is None:
                header = table[0]
                all_rows.extend(table[1:])
            else:
                all_rows.extend(table)
        if header is None:
            return None
        return pd.DataFrame(all_rows, columns=header)

if "model" not in st.session_state:
    st.session_state.model = SimpleModel()
if "data" not in st.session_state:
    st.session_state.data = []
if "trained" not in st.session_state:
    st.session_state.trained = False
st.title("School thingy")
step1, step2, step3 = st.tabs(["1- Upload", "2- Train", "3-Analyze"])
with step1:
    st.header("Upload your file")
    st.caption("for the model to learn")
    train_uploaded = st.file_uploader("Upload", type=["pdf"], key="train_pdf")
    if train_uploaded:
        train_df = extract_table_from_pdf(train_uploaded)
        if train_df is None or train_df.empty:
            st.error("No table found.")
        else:
            st.success(f"A table with {len(train_df)} rows in your table")
            st.write("Extracted Text")
            st.dataframe(train_df.head(8), use_container_width=True)
            st.write("Choose what each column represents")
            col_options = ["None"] + list(train_df.columns)
            train_mapping = {}
            m1,m2,m3 = st.columns(3)
            m4,m5,m6 = st.columns(3)
            train_mapping["attendance"] = m1.selectbox("Attendance", col_options, key="train_attendance")
            train_mapping["homework"] = m2.selectbox("Homework", col_options, key="train_homework")
            train_mapping["quiz"] = m3.selectbox("Quiz", col_options, key="train_quiz")
            train_mapping["arabic"] = m4.selectbox("Arabic", col_options, key="train_arabic")
            train_mapping["math"] = m5.selectbox("Math", col_options, key="train_math")
            train_mapping["english"] = m6.selectbox("English", col_options, key="train_english")
            outcome = st.selectbox("Results", col_options, key="train_results")
            ready = (all(train_mapping[f] != "None" for f in features) and outcome != "None")
            if not ready:
                st.warning("Choose all the required columns")
            else:
                if st.button("Use this data for training", use_container_width=True):
                    clean = pd.DataFrame()
                    for f in features:
                        clean[f] = pd.to_numeric(train_df[train_mapping[f]], errors="coerce")
                    clean["outcome"] = pd.to_numeric(train_df[outcome], errors="coerce")
                    clean = clean.dropna() #deletes incomplete data
                    st.session_state.data = (clean[features+["outcome"]].values.tolist())
                    st.success(f"{len(clean)} student's data is ready for training")
    if st.session_state.data:
        st.write(f"{len(st.session_state.data)} student's data is ready for training")
        cols = [features_ar[f] for f in features] + ["Final Result"]
        st.dataframe(pd.DataFrame(st.session_state.data, columns=cols), use_container_width=True, hide_index=True)
        if st.button("Clear the training data"):
            st.session_state.data = []
            st.session_state.trained = False
            st.session_state.model = SimpleModel()
            st.rerun()
with step2:
    st.header("Train the model")
    if len(st.session_state.data) < 8:
        st.warning("You need at least 8 students' data to train the model")
    else:
        if st.button("Train the model", type="primary", use_container_width=True):
            data = st.session_state.data
            x = [d[:6] for d in data]
            y = [d[6] for d in data]
            with st.spinner("Training in progress..."):
                st.session_state.model.train(x, y)
                st.session_state.trained = True
            st.success("Model trained successfully!")
            st.balloons()
        if st.session_state.trained:
            model = st.session_state.model
    if not st.session_state.trained:
        st.info("Train the model first")
with step3:
    st.header("Upload your pdf")
    if not st.session_state.trained:
        st.info("Train the model first in step 2")
    else:
        model = st.session_state.model
        uploaded = st.file_uploader("Upload your file", type=["pdf"])

        with st.spinner("Extracting..."):  
            df = extract_table_from_pdf(uploaded)
        if df is None or df.empty:
            st.error("No table found in the PDF.")
        else:
            st.success(f"A table with {len(df)} rows in your table")
            st.write("Data")
            st.dataframe(df.head(8), use_container_width=True)
            st.write("Choose each column")
            col_options = ["None"] + list(df.columns)
            mapping = {}
            m1,m2,m3 = st.columns(3)
            m4,m5,m6 = st.columns(3)
            name_col = m1.selectbox("اسم الطلاب", col_options, key="name_col")
            mapping["attendance"] = m2.selectbox("نسبة الحضور", col_options, key="attendance_col")
            mapping["homework"] = m3.selectbox("نسبة الواجب الدراسي", col_options, key="homework_col")
            mapping["quiz"] = m4.selectbox("نسبة الاختبار", col_options, key="quiz_col")
            mapping["arabic"] = m5.selectbox("عربي", col_options, key="arabic_col")
            mapping["math"] = m6.selectbox("رياضيات", col_options, key="math_col")
            mapping["english"] = st.selectbox("English", col_options, key="english_col")
            ready = all(mapping[f] != "None" for f in features)
            if not ready:
                st.warning("Choose all the required columns")
            else:
                if st.button("analyze", type="primary", use_container_width=True):
                    work = pd.DataFrame()
                if name_col != "None":
                    work["name"] = df[name_col]
                for f in features:
                    work[f] = pd.to_numeric(df[mapping[f]], errors="coerce")
                work = work.dropna(subset=features)
                work["final"] = model.predict_all(work[features].values).round(0)
                work["status"] = np.where(work["final"] >= 50, "Pass", "Fail")
                n = len(work)
                n_pass = int((work["final"] >= 50).sum())
                n_fail = n - n_pass
                n_ex = int((work["final"] >= 80).sum())
                st.divider()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Students", n)
                k2.metric("Pass", n_pass)
                k3.metric("Fail", n_fail)
                k4.metric("Excellent", n_ex)
                sub_avg ={
                    features_ar[f]: work[f].mean()
                    for f in ["arabic", "math", "english"]
                }
                weakest_subject = min(sub_avg, key=sub_avg.get)
                st.write(f"The weakest subject is: {weakest_subject} with an average of {sub_avg[weakest_subject]:.2f}")
                st.bar_chart(pd.Series(sub_avg))
                st.write("failed students")
                at_risk = work[work["final"] < 50].sort_values("final")
                if len(at_risk) == 0:
                    st.info("there's no student")
                else:
                    st.dataframe(at_risk, use_container_width=True, hide_index=True)
                st.write("result")
                st.dataframe(work.sort_values("final"), use_container_width=True, hide_index=True)