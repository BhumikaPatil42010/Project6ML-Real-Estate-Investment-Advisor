# # =========================
# # STREAMLIT APP
# # =========================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# st.set_page_config(page_title="Real Estate Advisor", layout="wide")

# st.title("Real Estate Investment Advisor")

# # =========================
# # LOAD MODELS
# # =========================

# base_dir = os.path.dirname(os.path.dirname(__file__))

# clf = joblib.load(os.path.join(base_dir, 'models', 'classifier.pkl'))
# reg = joblib.load(os.path.join(base_dir, 'models', 'regressor.pkl'))

# # =========================
# # USER INPUT FORM
# # =========================

# st.sidebar.header("Enter Property Details")

# bhk = st.sidebar.slider("BHK", 1, 5, 2)
# sqft = st.sidebar.number_input("Size (SqFt)", 500, 5000, 1000)
# price = st.sidebar.number_input("Current Price (Lakhs)", 10, 500, 50)
# floor = st.sidebar.slider("Floor No", 0, 50, 1)
# total_floors = st.sidebar.slider("Total Floors", 1, 60, 5)
# age = st.sidebar.slider("Property Age", 0, 30, 5)

# schools = st.sidebar.slider("Nearby Schools", 0, 10, 3)
# hospitals = st.sidebar.slider("Nearby Hospitals", 0, 10, 2)
# transport = st.sidebar.slider("Transport Access (1-Low,3-High)", 1, 3, 2)

# parking = st.sidebar.slider("Parking Spaces", 0, 5, 1)
# amenities_count = st.sidebar.slider("Amenities Count", 0, 10, 3)

# # Derived feature
# location_score = schools + hospitals + transport

# # =========================
# # CREATE INPUT DATAFRAME
# # =========================

# input_data = {
#     'bhk': bhk,
#     'size_in_sqft': sqft,
#     'price_per_sqft': (price * 100000) / sqft,
#     'year_built': 2026 - age,
#     'floor_no': floor,
#     'total_floors': total_floors,
#     'age_of_property': age,
#     'nearby_schools': schools,
#     'nearby_hospitals': hospitals,
#     'public_transport_accessibility': transport,
#     'parking_space': parking,
#     'amenities_count': amenities_count,
#     'location_score': location_score
# }

# # Convert to DataFrame
# input_df = pd.DataFrame([input_data])

# # =========================
# # MATCH TRAINING FEATURES
# # =========================

# # Load training columns
# train_cols_path = os.path.join(base_dir, 'models', 'classifier.pkl')
# model_features = clf.feature_names_in_

# for col in model_features:
#     if col not in input_df.columns:
#         input_df[col] = 0

# input_df = input_df[model_features]

# # =========================
# # PREDICTION
# # =========================

# if st.button("🔍 Predict"):

#     pred_class = clf.predict(input_df)[0]
#     pred_price = reg.predict(input_df)[0]

#     future_price = pred_price * (1.08 ** 5)

#     st.subheader("Results")

#     if pred_class == 1:
#         st.success("Good Investment")
#     else:
#         st.error("Not a Good Investment")

#     st.write(f"Estimated Price after 5 Years: ₹ {future_price:,.2f} Lakhs")

#     # =========================
#     # PRO FEATURE 1: Confidence Score
#     # =========================
#     prob = clf.predict_proba(input_df)[0][1]
#     st.write(f"Confidence Score: {prob:.2f}")

#     # =========================
#     #  PRO FEATURE 2: Feature Importance
#     # =========================
#     st.subheader("Top Important Features")

#     importances = clf.feature_importances_
#     feature_names = input_df.columns

#     feat_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': importances
#     }).sort_values(by='Importance', ascending=False).head(10)

#     st.bar_chart(feat_df.set_index('Feature'))
# # =========================
# # VISUAL INSIGHT (OPTIONAL)
# # =========================

# st.subheader(" Sample Insight")

# sample_data = pd.DataFrame({
#     'Feature': ['Schools', 'Hospitals', 'Transport'],
#     'Score': [schools, hospitals, transport]
# })

# st.bar_chart(sample_data.set_index('Feature'))

# # Email: bhumika@gmail.com
# #  streamlit run app/app.py

# ====================================================
# =========================
# ADVANCED STREAMLIT APP
# =========================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os

# st.set_page_config(page_title="Real Estate Advisor", layout="wide")

# # =========================
# # LOAD MODELS
# # =========================

# base_dir = os.path.dirname(os.path.dirname(__file__))

# clf = joblib.load(os.path.join(base_dir, 'models', 'classifier.pkl'))
# reg = joblib.load(os.path.join(base_dir, 'models', 'regressor.pkl'))

# # =========================
# # TITLE
# # =========================

# st.title("🏠 Real Estate Investment Advisor")
# st.markdown("### AI-based Property Investment Analysis")

# # =========================
# # SIDEBAR INPUTS
# # =========================

# st.sidebar.header("Enter Property Details")

# bhk = st.sidebar.slider("BHK", 1, 5, 2)
# sqft = st.sidebar.number_input("Size (SqFt)", 500, 5000, 1000)
# price = st.sidebar.number_input("Current Price (Lakhs)", 10, 500, 50)

# floor = st.sidebar.slider("Floor No", 0, 50, 1)
# total_floors = st.sidebar.slider("Total Floors", 1, 60, 5)
# age = st.sidebar.slider("Property Age", 0, 30, 5)

# schools = st.sidebar.slider("Nearby Schools", 0, 10, 3)
# hospitals = st.sidebar.slider("Nearby Hospitals", 0, 10, 2)
# transport = st.sidebar.slider("Transport Access", 1, 3, 2)

# parking = st.sidebar.slider("Parking Spaces", 0, 5, 1)
# amenities_count = st.sidebar.slider("Amenities Count", 0, 10, 3)

# location_score = schools + hospitals + transport

# # =========================
# # CREATE INPUT DATA
# # =========================

# input_data = {
#     'bhk': bhk,
#     'size_in_sqft': sqft,
#     'year_built': 2026 - age,
#     'floor_no': floor,
#     'total_floors': total_floors,
#     'age_of_property': age,
#     'nearby_schools': schools,
#     'nearby_hospitals': hospitals,
#     'public_transport_accessibility': transport,
#     'parking_space': parking,
#     'amenities_count': amenities_count,
#     'location_score': location_score
# }

# input_df = pd.DataFrame([input_data])

# # Match training features
# model_features = clf.feature_names_in_

# for col in model_features:
#     if col not in input_df.columns:
#         input_df[col] = 0

# input_df = input_df[model_features]

# # =========================
# # TABS
# # =========================

# tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Insights"])

# # =========================
# # TAB 1: PREDICTION
# # =========================

# with tab1:

#     if st.button("Predict Now"):

#         pred_class = clf.predict(input_df)[0]
#         pred_price = reg.predict(input_df)[0]
#         future_price = pred_price * (1.08 ** 5)

#         col1, col2, col3 = st.columns(3)

#         # Investment result
#         with col1:
#             if pred_class == 1:
#                 st.metric("Investment Decision", "Good ✅")
#             else:
#                 st.metric("Investment Decision", "Not Good ❌")

#         # Price prediction
#         with col2:
#             st.metric("Future Price (5 yrs)", f"₹ {future_price:,.2f} L")

#         # Confidence
#         with col3:
#             prob = clf.predict_proba(input_df)[0][1]
#             st.metric("Confidence", f"{prob:.2f}")

#         # =========================
#         # Feature Importance
#         # =========================

#         st.subheader("Top Influencing Features")

#         importances = clf.feature_importances_

#         feat_df = pd.DataFrame({
#             'Feature': model_features,
#             'Importance': importances
#         }).sort_values(by='Importance', ascending=False).head(10)

#         st.bar_chart(feat_df.set_index('Feature'))

# # =========================
# # TAB 2: INSIGHTS
# # =========================

# with tab2:

#     st.subheader("Property Score Breakdown")

#     score_df = pd.DataFrame({
#         'Feature': ['Schools', 'Hospitals', 'Transport'],
#         'Score': [schools, hospitals, transport]
#     })

#     st.bar_chart(score_df.set_index('Feature'))

#     st.subheader("Quick Analysis")

#     if location_score > 10:
#         st.success("Excellent Location")
#     elif location_score > 6:
#         st.warning("Moderate Location")
#     else:
#         st.error("Low Location Score")

#     if amenities_count > 5:
#         st.success("High Amenities")
#     else:
#         st.info("Basic Amenities")

# # =========================
# # FOOTER
# # =========================

# st.markdown("---")
# st.markdown("Built with ❤️ using Streamlit | ML Project")

# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Real Estate Advisor", layout="wide")

# =========================
# LOAD MODELS + DATA
# =========================

base_dir = os.path.dirname(os.path.dirname(__file__))

clf = joblib.load(os.path.join(base_dir, 'models', 'classifier.pkl'))
reg = joblib.load(os.path.join(base_dir, 'models', 'regressor.pkl'))

df = pd.read_csv(os.path.join(base_dir, 'data', 'cleaned_data.csv'))

# =========================
# TITLE
# =========================

st.title("🏠 Real Estate Investment Advisor")
st.markdown("### Smart AI-based Property Analysis System")

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("Filter Properties")

selected_bhk = st.sidebar.selectbox("BHK", [1,2,3,4,5])
max_price = st.sidebar.slider("Max Price (Lakhs)", 10, 500, 100)
min_sqft = st.sidebar.slider("Min Size (SqFt)", 500, 5000, 800)

filtered_df = df[
    (df['BHK'] == selected_bhk) &
    (df['Price_in_Lakhs'] <= max_price) &
    (df['Size_in_SqFt'] >= min_sqft)
]

st.subheader("Filtered Properties")
st.write(filtered_df.head())

# =========================
# USER INPUT FORM
# =========================

st.sidebar.header("Enter Property Details")

bhk = st.sidebar.slider("BHK", 1, 5, 2)
sqft = st.sidebar.number_input("Size", 500, 5000, 1000)
price = st.sidebar.number_input("Current Price (Lakhs)", 10, 500, 50)

floor = st.sidebar.slider("Floor No", 0, 50, 1)
total_floors = st.sidebar.slider("Total Floors", 1, 60, 5)
age = st.sidebar.slider("Property Age", 0, 30, 5)

schools = st.sidebar.slider("Nearby Schools", 0, 10, 3)
hospitals = st.sidebar.slider("Nearby Hospitals", 0, 10, 2)
transport = st.sidebar.slider("Transport (1-3)", 1, 3, 2)

parking = st.sidebar.slider("Parking", 0, 5, 1)
amenities_count = st.sidebar.slider("Amenities Count", 0, 10, 3)

location_score = schools + hospitals + transport

# =========================
# INPUT DATAFRAME
# =========================

input_data = {
    'bhk': bhk,
    'size_in_sqft': sqft,
    'year_built': 2026 - age,
    'floor_no': floor,
    'total_floors': total_floors,
    'age_of_property': age,
    'nearby_schools': schools,
    'nearby_hospitals': hospitals,
    'public_transport_accessibility': transport,
    'parking_space': parking,
    'amenities_count': amenities_count,
    'location_score': location_score
}

input_df = pd.DataFrame([input_data])

# Match training features
model_features = clf.feature_names_in_

for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    pred_class = clf.predict(input_df)[0]
    pred_price = reg.predict(input_df)[0]
    future_price = pred_price * (1.08 ** 5)

    col1, col2, col3 = st.columns(3)

    # Classification
    with col1:
        if pred_class == 1:
            st.success("Good Investment")
        else:
            st.error("Not a Good Investment")

    # Regression
    with col2:
        st.metric("Future Price (5 yrs)", f"₹ {future_price:,.2f} Lakhs")

    # Confidence
    with col3:
        prob = clf.predict_proba(input_df)[0][1]
        st.metric("Confidence", f"{prob:.2f}")

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    st.subheader("Feature Importance")

    importances = clf.feature_importances_

    feat_df = pd.DataFrame({
        'Feature': model_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    st.bar_chart(feat_df.set_index('Feature'))

# =========================
# VISUAL INSIGHTS
# =========================

st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

# Price vs Size
with col1:
    fig, ax = plt.subplots()
    sns.scatterplot(x='Size_in_SqFt', y='Price_in_Lakhs', data=df.sample(5000), ax=ax)
    ax.set_title("Size vs Price")
    st.pyplot(fig)

# Heatmap
with col2:
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# =========================
# LOCATION INSIGHT
# =========================

st.subheader("Location Score Analysis")

score_df = pd.DataFrame({
    'Factor': ['Schools', 'Hospitals', 'Transport'],
    'Score': [schools, hospitals, transport]
})

st.bar_chart(score_df.set_index('Factor'))

# “The application allows users to filter properties, input custom data, and get real-time predictions along with confidence scores and visual insights like heatmaps and trend charts.”