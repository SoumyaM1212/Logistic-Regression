
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from cleaning import clean_dataset


st.title("Fish Species Prediction")

file = st.file_uploader("Upload Fish CSV", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    # Save original fish names
    fish_names = sorted(df["Species"].dropna().unique())

    st.subheader("Complete Data")
    st.dataframe(df)

    # Clean data
    d = clean_dataset(df.copy())

    st.subheader("Cleaned Data")
    st.dataframe(d)

    # Features and target
    X = d[["Length3", "Height", "Width"]]
    Y = d["Species"]

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # Accuracy
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)

    st.subheader("Model Accuracy")
    st.write(accuracy * 100, "%")

    # Fish inputs
    st.subheader("Predict Fish Species")

    length = st.number_input("Length3", min_value=0.0, value=30.0)
    height = st.number_input("Height", min_value=0.0, value=10.0)
    width = st.number_input("Width", min_value=0.0, value=5.0)

    if st.button("Predict"):

        input_data = pd.DataFrame(
            [[length, height, width]],
            columns=["Length3", "Height", "Width"]
        )

        result = model.predict(input_data)[0]

        # Convert predicted number to fish name
        fish_name = fish_names[int(result)]

        st.success("Predicted Fish Species: " + fish_name)

