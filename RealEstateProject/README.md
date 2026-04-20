
# 🏠 Real Estate Investment Advisor

## 📌 Project Overview

This project is a Machine Learning-based application designed to help users make informed real estate investment decisions. It predicts:

* ✅ Whether a property is a **Good Investment** (Classification)
* 💰 Estimated **Property Price after 5 Years** (Regression)

The system also provides an interactive dashboard built using Streamlit for real-time predictions and insights.

---

## 🎯 Objectives

* Analyze real estate data to identify investment patterns
* Build ML models for classification and regression
* Provide a user-friendly interface for predictions
* Enable data-driven decision-making for investors

---

## 📊 Dataset Description

The dataset contains property details such as:

* Location (State, City)
* Property Type (Apartment, Villa, House)
* BHK (Bedrooms, Hall, Kitchen)
* Size (SqFt)
* Price (Lakhs)
* Amenities, Parking, Security
* Nearby Schools & Hospitals
* Transport Accessibility

---

## 🔧 Technologies Used

* Python 🐍
* Pandas, NumPy
* Scikit-learn
* Random Forest Algorithm
* Streamlit (UI)
* MLflow (Experiment Tracking)
* Matplotlib / Seaborn (EDA)

---

## 🧪 Machine Learning Approach

### 🔹 Data Preprocessing

* Handled missing values
* Removed duplicates
* Converted categorical features to numerical
* Removed irrelevant and leakage features

---

### 🔹 Feature Engineering

* Created:

  * `location_score`
  * `amenities_count`
* Dropped high-correlation features like `price_per_sqft`

---

### 🔹 Model Training

#### 📌 Classification Model

* Algorithm: Random Forest Classifier
* Target: `good_investment`
* Metrics:

  * Accuracy: ~81%
  * F1 Score: ~0.70

#### 📌 Regression Model

* Algorithm: Random Forest Regressor
* Target: `price_in_lakhs`
* Metrics:

  * RMSE: Low
  * R² Score: Good

---

## 📈 Results

* Achieved realistic prediction performance
* Eliminated data leakage issues
* Built stable and scalable ML pipeline

---

## 🖥️ Streamlit Application Features

* Interactive user input form
* Investment prediction (Good / Not Good)
* Future price estimation (5 years)
* Confidence score display
* Feature importance visualization
* Insights dashboard with charts

---

## 🚀 How to Run the Project

### 🔹 Step 1: Clone Repository

```bash
git clone https://github.com/your-username/RealEstateProject.git
cd RealEstateProject
```

### 🔹 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 3: Train Model

```bash
python src/train.py
```

### 🔹 Step 4: Run Streamlit App

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
RealEstateProject/
│
├── data/
├── models/
├── src/
├── app/
├── notebooks/
├── mlruns/
├── requirements.txt
└── README.md
```

---

## 📌 Key Learnings

* Importance of feature engineering
* Handling data leakage in ML models
* Model evaluation using proper metrics
* Building end-to-end ML applications
* Creating interactive dashboards

---

## 🔮 Future Improvements

* Use XGBoost for better performance
* Add location-based map visualization
* Deploy on cloud (Streamlit Cloud / AWS)
* Integrate real-time property data APIs

---

## 👩‍💻 Author

**Bhumika Patil**

---

## ⭐ Acknowledgements

* Open-source ML libraries
* Streamlit documentation
* Dataset inspiration from real estate platforms

---


