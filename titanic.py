import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from data_cleaning import clean_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
st.title("Titanic")
 
upload_file = st.file_uploader("Upload Titanic CSV Dataset",type=["csv"])
if upload_file is not None:
    df = pd.read_csv(upload_file)

    tab1,tab2,tab3,tab4,tab5,tab6= st.tabs(["Complete Data","Describe","Clean Data","Top 5 rows","Bottom 5 rows","Analysis of Data"])
 
    with tab1:
        st.write("Dataset")
        st.dataframe(df)
 
    with tab2:
        st.write("Describe")
        st.dataframe(df.describe())
 
 
    with tab3:
        st.write("Clean Data")
        cleaned_df = clean_dataset(df.copy())
        st.dataframe(cleaned_df,use_container_width=True)
 
  
    with tab4:
        st.write("Top 5 rows")
        st.dataframe(cleaned_df.head())
 
    with tab5:
        st.write("Bottom 5 rows")
        st.dataframe(cleaned_df.tail())

    ml_df = df.copy()
    ml_df["Sex"] = ml_df["Sex"].map({
    "male": 0,
    "female": 1})

    X =  ml_df[["Pclass","Sex", "Age", "SibSp", "Parch", "Fare"]]
 
    Y = ml_df["Survived"]

    X["Age"] = X["Age"].fillna(X["Age"].mean())
    X["Fare"] = X["Fare"].fillna(X["Fare"].mean())
    X["Sex"] = X["Sex"].fillna(X["Sex"].mode()[0])

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
 
 
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
 
    accuracy = accuracy_score(Y_test, pred)

    st.write("Accuracy:", accuracy)

    st.write("Predicted Values:")

    st.write(pred)

    st.write("Prediction Probability:")

    st.write(proba)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(Y_test,pred)

    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Not Survived", "Survived"])
    ax.set_yticklabels(["Not Survived", "Survived"])

    for i in range(2):
        for j in range(2):
            ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

    st.pyplot(fig)