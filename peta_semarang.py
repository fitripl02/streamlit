import streamlit as st
import pandas as pd
import pydeck as pdk

df = pd.read_csv("semarang_resto_dataset.csv")

st.header("Peta Sebaran Restoran di Semarang")
st.map(df[['latitude', 'longitude']])
