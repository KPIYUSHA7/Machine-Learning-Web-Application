#BREAST CANCER CLASSIFICATION DATASET : WEB APPLICATION 

import pandas as pd
import streamlit as st
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

#Read the data
data = pd.read_csv("data/data.csv")

#Process Data to Split Into target and features
y = data.diagnosis                           
drop_cols = ['Unnamed: 32','id','diagnosis']
x = data.drop(drop_cols,axis = 1 )
x.head()

#Drop the correlated features
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean',
              'concave points_mean','radius_se','perimeter_se',
              'radius_worst','perimeter_worst','compactness_worst',
              'concave points_worst','compactness_se','concave points_se',
              'texture_worst','area_worst']

x_1 = x.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 
x_1.head()

# print("Columns", x_1.columns)

#st.title("Classification On the Breast Cancer Dataset")
st.markdown("<h1 style='text-align: center; color: red;'>Breast Cancer Dataset : A Classification Problem</h1>", unsafe_allow_html=True)

#Add the logo
from PIL import Image
image = Image.open('logo.png')
st.image(image, caption='Breast Cancer Awareness', use_column_width=False, width=40,     )

st.sidebar.header("Variable Clinical Parameters \n")
st.sidebar.header("_M (Mean) _SE (Std. Error) _W (Worst)")

def user_input_features():
    texture_mean = st.sidebar.slider('Texture_M', 9.71, 39.28, 16.17)
    area_mean = st.sidebar.slider('Area_M', 143.5, 2501.01, 420.3)
    smoothness_mean = st.sidebar.slider('Smoothness_M', 0.05263, 0.1634, 0.08637)
    concavity_mean = st.sidebar.slider('Concavity_M', 0.01, 0.4268, 0.02956)
    symmetry_mean = st.sidebar.slider('Symmetry_M', 0.106, 0.304, 0.1619)
    fractal_dimension_mean = st.sidebar.slider('Fractal Dimension_M', 0.04996, 0.09744, 0.06154)
    texture_se = st.sidebar.slider('Texture_SE', 0.3602, 4.885, 0.8339)
    area_se = st.sidebar.slider('Area_SE', 6.802, 542.2, 24.53)
    smoothness_se = st.sidebar.slider('Smoothness_SE', 0.001713, 0.031130, 0.005169)
    concavity_se = st.sidebar.slider('Concavity_SE', 0.01, 0.396, 0.015090)
    symmetry_se = st.sidebar.slider('Symmetry_SE', 0.007882, 0.078950, 0.023480)
    fractal_dimension_se = st.sidebar.slider('Fractal Dimension_SE', 0.000895, 0.029840, 0.004558)
    smoothness_worst = st.sidebar.slider('Smoothness_W', 0.071170, 0.222600, 0.146000)
    concavity_worst = st.sidebar.slider('Concavity_W', 0.01, 1.252, 0.3829)
    symmetry_worst = st.sidebar.slider('Symmetry_W', 0.1565, 0.6638, 0.3179)
    fractal_dimension_worst = st.sidebar.slider('Fractal_Dimension_W', 0.055040, 0.207500, 0.092080)
    
    data = {'texture_mean': texture_mean,
            'area_mean': area_mean,
            'smoothness_mean': smoothness_mean,
            'concavity_mean': concavity_mean,
           'symmetry_mean': symmetry_mean,
           'fractal_dimension_mean': fractal_dimension_mean,
           'texture_se': texture_se,
           'area_se': area_se,
           'smoothness_se': smoothness_se,
            'concavity_se': concavity_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'smoothness_worst': smoothness_worst,
            'concavity_worst': concavity_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst
           }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader("Input Parameters")
st.write(df)

X = x_1
y = y

#Train an XGBoost Classifier 
clf = xgb.XGBClassifier()
clf.fit(x_1, y)

#Predict whether sample is having Malignant or Benign Breast Cancer
prediction = clf.predict(df)

prediction_probability = clf.predict_proba(df)

st.subheader('Prediction')
st.subheader(prediction)

st.subheader("Assignment Of Class Labels According to probability \n")
a = st.slider('Probability', 0.00, 1.00, 0.5)
if a == 0:
    st.write("Benign Sample")
elif a == 1:
    st.write("Malignant Sample")
else:
    st.write("Outcome is the label of class belonging to higher probability")

st.subheader('Prediction Probability')
st.write(prediction_probability)

