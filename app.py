import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Smart Retail System", layout="wide")

# ---------- Custom Styling ----------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
h1, h2, h3 {
    color: #1f4e79;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Dataset", "🤖 Models", "📈 Visualization"])

# ---------- Data Generator ----------
@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        "Age": np.random.randint(18, 65, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Annual Income": np.random.randint(20000, 120000, n),
        "Purchase Frequency": np.random.randint(1, 20, n),
        "Discount Offered": np.random.randint(0, 50, n),
        "Product Category": np.random.choice(["Electronics", "Clothing", "Grocery"], n),
        "Marketing Spend": np.random.randint(1000, 20000, n),
        "Seasonal Demand Index": np.random.uniform(0.5, 1.5, n),
        "Store Location": np.random.choice(["Urban", "Semi-Urban", "Rural"], n)
    })

    data["Monthly Sales"] = (
        data["Annual Income"] * 0.05 +
        data["Purchase Frequency"] * 200 +
        data["Marketing Spend"] * 0.1 +
        np.random.normal(0, 1000, n)
    )

    data["Purchase Decision"] = np.where(data["Discount Offered"] > 20, 1, 0)

    data["Loyalty Category"] = pd.cut(
        data["Purchase Frequency"],
        bins=[0, 5, 12, 20],
        labels=["Low", "Medium", "High"]
    )

    return data


# ================= HOME =================
if page == "🏠 Home":
    st.title("🛍 Smart Retail Sales Forecasting & Customer Segmentation")
    st.markdown("### 🚀 INSIGHTX 2026 Hackathon Project")

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithms Used", "4")
    col2.metric("Dataset Size", "1000+ Records")
    col3.metric("System Type", "ML Web App")

    st.markdown("---")
    st.info("""
    🔹 Linear Regression → Predict Monthly Sales  
    🔹 Decision Tree → Purchase Decision  
    🔹 KNN → Loyalty Prediction  
    🔹 K-Means → Customer Segmentation  
    """)

    st.success("Team: Team II")


# ================= DATASET =================
elif page == "📊 Dataset":

    st.title("📊 Dataset Overview")
    data = generate_data()

    col1, col2 = st.columns(2)
    col1.metric("Total Customers", len(data))
    col2.metric("Average Income", f"₹ {int(data['Annual Income'].mean())}")

    st.dataframe(data.head())


# ================= MODELS =================
elif page == "🤖 Models":

    st.title("🤖 Machine Learning Performance Dashboard")

    data = generate_data()

    # Encode categorical
    le = LabelEncoder()
    for col in ["Gender", "Product Category", "Store Location", "Loyalty Category"]:
        data[col] = le.fit_transform(data[col])

    X = data.drop(["Monthly Sales", "Purchase Decision", "Loyalty Category"], axis=1)

    # ---------- Linear Regression ----------
    y_reg = data["Monthly Sales"]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train_reg, y_train_reg)
    y_pred_reg = lr.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)

    # ---------- Decision Tree ----------
    y_dt = data["Purchase Decision"]
    X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
        X, y_dt, test_size=0.2, random_state=42
    )

    dt = DecisionTreeClassifier()
    dt.fit(X_train_dt, y_train_dt)
    y_pred_dt = dt.predict(X_test_dt)
    acc_dt = accuracy_score(y_test_dt, y_pred_dt)

    # ---------- KNN ----------
    y_knn = data["Loyalty Category"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X_scaled, y_knn, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_knn, y_train_knn)
    y_pred_knn = knn.predict(X_test_knn)
    acc_knn = accuracy_score(y_test_knn, y_pred_knn)

    # ---------- KPI Cards ----------
    st.markdown("### 📊 Model Performance Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("📈 Linear Regression MSE", f"{mse:.2f}")
    c2.metric("🌳 Decision Tree Accuracy", f"{acc_dt:.3f}")
    c3.metric("👥 KNN Accuracy", f"{acc_knn:.3f}")

    st.markdown("---")

    # ---------- Tabs ----------
    tab1, tab2, tab3 = st.tabs(["📈 Linear Regression", "🌳 Decision Tree", "👥 KNN"])

    with tab1:
        st.subheader("Actual vs Predicted Sales")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test_reg, y_pred_reg, alpha=0.6)
        ax1.set_xlabel("Actual Sales")
        ax1.set_ylabel("Predicted Sales")
        st.pyplot(fig1)

    with tab2:
        st.subheader("Decision Tree Confusion Matrix")
        cm = confusion_matrix(y_test_dt, y_pred_dt)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        st.pyplot(fig2)
        st.text(classification_report(y_test_dt, y_pred_dt))

    with tab3:
        st.subheader("KNN Confusion Matrix")
        cm_knn = confusion_matrix(y_test_knn, y_pred_knn)
        fig3, ax3 = plt.subplots()
        sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Greens", ax=ax3)
        st.pyplot(fig3)
        st.text(classification_report(y_test_knn, y_pred_knn))


# ================= VISUALIZATION =================
elif page == "📈 Visualization":

    st.title("📊 Data Visualization & Clustering")

    data = generate_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.select_dtypes(include=np.number))

    kmeans = KMeans(n_clusters=3, random_state=42)
    data["Cluster"] = kmeans.fit_predict(X_scaled)

    st.subheader("🟢 Customer Segmentation")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=data,
                    x="Annual Income",
                    y="Purchase Frequency",
                    hue="Cluster",
                    palette="Set2",
                    ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Sales Distribution")
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.histplot(data["Monthly Sales"], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)