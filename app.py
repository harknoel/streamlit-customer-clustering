import streamlit as st
import pandas as pd
import pickle

# Load the pickled KMeans model
filename = 'rfm_kmeans_model.pkl'
with open(filename, 'rb') as infile:
    loaded_model = pickle.load(infile)

st.title('RFM Customer Segmentation')

# Get user input for RFM values
recency = st.number_input('Recency (Days since last purchase)', min_value=0)
frequency = st.number_input('Frequency (Number of purchases)', min_value=0)
monetary = st.number_input('Monetary (Total spent)', min_value=0.0, format="%.3f", step=0.001)

# Prepare user input as a DataFrame
user_data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary], 'Clusters': [0]})

if st.button('Predict Cluster'):
    # Predict cluster label for user input
    prediction = loaded_model.predict(user_data)
    cluster = prediction[0]

    # Display prediction result
    st.write(f"Predicted Cluster: {cluster}")

    # Optionally, display cluster descriptions based on the model
    if cluster == 0:
        st.write("Description for Cluster 0 (e.g., Loyal High Spenders)")
    elif cluster == 1:
        st.write("Description for Cluster 1 (e.g., Recent Customers)")
    else:
        st.write("Description for Cluster 2 (e.g., Low Engagement Customers)")