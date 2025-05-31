import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("semarang_resto_dataset.csv")

st.header("Visualisasi Data Restoran")

fig, ax = plt.subplots()
sns.histplot(df["rating"], bins=10, kde=True, ax=ax)
st.pyplot(fig)
