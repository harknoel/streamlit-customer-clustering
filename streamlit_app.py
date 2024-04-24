import streamlit as st
import pandas as pd
import pickle

# Load the pickled KMeans model
filename = 'rfm_kmeans_model.pkl'
with open(filename, 'rb') as infile:
    loaded_model = pickle.load(infile)

st.title('RFM Customer Segmentation')

string = """
```
Customers in Cluster 1 are classified as best customers.
Customers in Cluster 2 are classified as worst customers.
```
"""

st.markdown(string, unsafe_allow_html=True)

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

    st.write(f"Predicted Cluster: {cluster}")

    # Display cluster descriptions based on the model
    cluster_0 = """
    1. They are less frequent
    2. They make purchases of small amounts
    3. They have made more recent purchases
    """

    cluster_1 = """
    1. They frequently make purchases
    2. They make purchases of large amounts
    3. They have made purchases more recently than customers in Cluster 0
    """

    cluster_2 = """
    1. They are less frequent
    2. They make purchases of small amounts
    3. They have not made a purchase in a long time.
    """

    if cluster == 0:
        st.write(cluster_0)
    elif cluster == 1:
        st.write(cluster_1)
    elif cluster == 2:
        st.write(cluster_2)
