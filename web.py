#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import streamlit as st
import statsmodels.formula.api as smf
from pickle import load

st.title("Prediction For Car Mileage")
st.sidebar.header("Input your parameters:")
def user_input():
    VOL=st.sidebar.number_input("Enter a Volume:")
    SP=st.sidebar.number_input("Enter a SP:")
    HP=st.sidebar.number_input("Enter a HP:")
    data={"VOL":VOL,
          "SP":SP,
          "log_hp":np.log(HP)}
    features=pd.DataFrame(data,index=[0])
    return features

df=user_input()
st.subheader("your input parameters")
st.write(df)

#load the model
loaded_model=load(open("finalized_model.pkl",'rb'))
prediction=loaded_model.predict(df)

st.subheader("The predicted Mileage:")
st.write(prediction)

