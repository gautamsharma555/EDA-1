import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set page config
st.set_page_config(page_title="Cardiotocographic EDA", layout="wide")

# Load dataset
st.title("Cardiotocographic Data - Exploratory Data Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("First 10 Rows of Data")
    st.dataframe(data.head(10))

    # Dataset Info
    st.subheader("Data Info")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Basic Shape and Type Info
    st.subheader("Dataset Dimensions and Types")
    st.write(f"Shape: {data.shape}")
    st.write(f"Number of Columns: {data.shape[1]}")
    st.write("Data Types:")
    st.write(data.dtypes)

    # Summary Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe())

    # Duplicates
    st.subheader("Duplicate Rows")
    st.write("Total Duplicate Rows:", data[data.duplicated()].shape[0])
    data = data.drop_duplicates()
    st.success("Duplicates removed")

    # Null Values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(data.isnull(), cbar=False, cmap=["green", "red"], ax=ax)
    st.pyplot(fig)

    # Impute missing values with median
    st.subheader("Imputing Missing Values with Median")
    data = data.fillna(data.median(numeric_only=True))
    st.success("Missing values imputed successfully")

    # Outlier treatment using IQR
    st.subheader("Outlier Treatment")
    for col in data.select_dtypes(include=np.number).columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data[col] = np.where(data[col] > upper_bound, upper_bound,
                             np.where(data[col] < lower_bound, lower_bound, data[col]))
    st.success("Outliers treated using IQR method")

    # Boxplot after outlier treatment
    st.subheader("Boxplots After Outlier Treatment")
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(data[col], ax=ax)
        ax.set_title(f"Boxplot for {col}")
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Pairplot (optional - can be heavy)
    if st.checkbox("Show Pairplot (May take time)"):
        st.subheader("Pairplot")
        fig = sns.pairplot(data)
        st.pyplot(fig)

    # Summary
    st.subheader("Summary")
    def summarize_findings():
        st.markdown("""
        ### Key Insights:
        1. Data cleaning was completed by removing duplicates and imputing missing values using median.
        2. Outliers were treated using IQR-based capping.
        3. Boxplots, correlation heatmap, and optional pairplot provide insights into feature distributions and relationships.
        4. The data is now ready for model building or further statistical analysis.
        """)

    summarize_findings()
