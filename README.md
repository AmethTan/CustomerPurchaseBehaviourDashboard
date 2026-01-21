# 🛒 E-commerce Customer Purchase Behaviour Analytic Dashboard (CusPurBeAD v1.0)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cuspurbead.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-ADFCBv-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**A Machine Learning-powered dashboard designed to democratise predictive analytics for Small and Medium Enterprises (SMEs).**

This repository contains the source code and resources for the **CusPurBeAD** application, developed as part of the Final Year Project for the Bachelor of Science Honours (Decision Science) at Universiti Utara Malaysia (UUM).

---

## 🔗 Quick Links
* **Live Dashboard:** [Click Here to Launch App](https://cuspurbead.streamlit.app/)
* **Full Project Report:** [Read the Research Paper](https://docs.google.com/document/d/1xEQcgb1MkOziozqrFKpNxYd5anKA5mjP/edit?usp=sharing&ouid=108260402295670215533&rtpof=true&sd=true)

---

## 📝 Project Overview

The rapid expansion of the digital marketplace has created an "information overload" paradox. While customer data is abundant, SMEs often struggle to extract actionable intelligence from transaction logs due to the complexity of "black box" machine learning models.

**CusPurBeAD** addresses this by providing an accessible, interactive interface that translates complex ML predictions into strategic insights. The project follows the **ADFCBv Lifecycle** (Aim, Data, Fix, Core, Board, Valid) to standardise the transformation of raw e-commerce data into revenue-optimising decisions.

### 🎯 Research Objectives
This project was guided by four primary objectives:
1. To identify and categorise the factors and dimensions of e-commerce consumer purchase behaviour.
2. To evaluate and select the optimal baseline machine learning models.
3. To formulate a comprehensive development framework for building a customer purchase behaviour analytical dashboard.
4. To develop, test and validate a functional dashboard that enables marketers to improve decision-making.

---

## 🧠 Machine Learning Core

The dashboard integrates a hybrid stack of three specific machine learning models, validated using a large-scale electronics dataset:

| Model | Application | Performance Metric | Key Insight |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Purchase Prediction | **ROC-AUC: 0.979** | Confirmed that dynamic metrics (like cart abandonment) are crucial predictors. |
| **FP-Growth** | Association Rules | **Lift > 9000** | Successfully mined "long-tail" product bundles often missed by standard analysis. |
| **SASRec** | Sequential Recommendation | **Recall@10: 86.8%** | Effectively modelled sequential user intent to predict the next likely interaction. |

---

## 💻 Dashboard Features & Usage

The application is built using **Streamlit** and is divided into three main analytical modules:

### 1. Purchase Prediction
Adjust customer and product profiles to gauge conversion probability in real-time.
* **Customer Inputs:** Recency, Frequency, Monetary (AOV/CLV), Time on Page, Abandonment Rate.
* **Product Inputs:** Category, Brand, Price, Popularity.

### 2. Smart Recommendations
Utilising FP-Growth, the system suggests products frequently bought together based on the current selection.

### 3. Clickstream Simulation
Using SASRec, the dashboard simulates potential future browsing paths based on the user's session history.

### How to Use
1.  **Sidebar Configuration:** Input the customer's behavioural metrics (e.g., Days since last visit, Avg pages/session).
2.  **Product Interaction:** Select a product category and brand to simulate a user viewing an item.
3.  **Analyze Results:**
    * Check the **Purchase Probability Gauge**.
    * Review **Smart Recommendations** for cross-selling opportunities.
    * Add items to **Session History** to see predicted next steps in the simulation engine.

---

## 📂 Dataset Details

The models were trained on the **"eCommerce events history in electronics store"** dataset (Oct 2019 – Feb 2020), sourced via Kaggle from [Rees46 Marketing Platform](https://rees46.com/) by Michael Kechinov.

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store)
* **Size:** Large-scale transaction logs from a multi-category store.

**Key Features Used:**
* `event_type`: view, cart, remove_from_cart, purchase.
* `product_id` / `category_code`: Hierarchical product data.
* `brand` / `price`: Product attributes.
* `user_session`: Used to reconstruct user journeys.

---

## 🛠 Installation (Local)

To run this dashboard locally on your machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AmethTan/CustomerPurchaseBehaviourDashboard.git](https://github.com/AmethTan/CustomerPurchaseBehaviourDashboard.git)
    cd CustomerPurchaseBehaviourDashboard
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run dashboard.py
    ```

---

## 🎓 Academic Details

**University:** Universiti Utara Malaysia (UUM)
**School:** School of Quantitative Sciences (SQS)
**Program:** Bachelor Degree of Science Honours (Decision Science)

* **Student Name:** Tan Yu Xian
* **Matric Number:** 292083
* **Supervisor:** Dr. Mohd Aamir Adeeb bin Abdul Rahim

---

## 📜 License

This project is for educational and academic purposes.
[MIT License](LICENSE)

***
*Developed with ❤️ using [Streamlit](https://streamlit.io)*
