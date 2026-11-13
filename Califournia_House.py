'''
Problem :California House Rent Price Prediction
Algorithm : KNN Classification 

'''
import streamlit as st
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

st.title("California House Price Prediction using KNN")

data = fetch_california_housing(as_frame=True)
df = data.frame

st.subheader("Complete Dataset")
st.dataframe(df)


df = df.drop_duplicates()
df = df.dropna()

st.subheader("Cleaned Dataset")
st.dataframe(df)

X = df.drop("MedHouseVal", axis=1)
Y = df["MedHouseVal"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, Y_train)

pred = model.predict(X_test)

accuracy = r2_score(Y_test, pred)

st.subheader("Model Accuracy")
st.write(round(accuracy * 100, 2), "%")

st.subheader("Enter House Details")

longitude = st.number_input("Longitude", value=-122.0)
latitude = st.number_input("Latitude", value=37.0)
housing_median_age = st.number_input("Housing Median Age", value=30.0)
total_rooms = st.number_input("Total Rooms", value=2000.0)
total_bedrooms = st.number_input("Total Bedrooms", value=400.0)
population = st.number_input("Population", value=1000.0)
households = st.number_input("Households", value=350.0)
median_income = st.number_input("Median Income", value=4.0)

# PREDICTION
if st.button("Predict House Price"):

    values = [[ longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income]]

    values = scaler.transform(values)

    result = model.predict(values)

    st.success("Predicted House Price:",str(round(result[0] * 100000, 2)))

    distances, indexes = model.kneighbors(values)

    st.subheader("Five Nearest Neighbours")

    for i in indexes[0]:
        st.write("House Price:",round(Y_train.iloc[i] * 100000, 2))
