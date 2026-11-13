'''
Problem: Credit card fraud detection 
Algorithm : Logistic Regression

'''



import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix ,roc_curve, auc
from data_cleaning import clean_dataset

st.title("Credit Card Fraud Detection")

file = st.file_uploader("Upload Credit Card Dataset", type=["csv"])
 
if  file is not None:
    df = pd.read_csv(file)

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

 
    data = df.copy()
    X = data.drop("Class", axis=1)
    Y = data["Class"]

    X = X.fillna(X.mean())

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, Y_train)

    pred = model.predict(X_test)

    st.subheader("Prediction Results")

    st.write("Accuracy:", accuracy_score(Y_test, pred))

    st.write(pred[:10]) 

    # CONFUSION MATRIX

    cm = confusion_matrix(Y_test, pred)

    print("Confusion Matrix:")
    print(cm)
    plt.imshow(cm, cmap="Blues")
    plt.title("KNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0, 1], ["Normal", "Fraud"])
    plt.yticks([0, 1], ["Normal", "Fraud"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.show()

    # ROC CURVE
    proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, threshold = roc_curve(Y_test, proba)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"KNN AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("KNN ROC Curve")

    plt.legend()
    plt.show()

    print("AUC Score:", round(roc_auc, 3))

    