#  Churn Prediction & Segmentation For Retention Strategy For Ecommerce | Machine Learning - Python

![image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/fd628395ff011ba158a382848dd5d9b8a3a5083c/documents/1_5vzYeD8PE6DORWP72Wv71w.png)

**Author:** Lê Anh Tuấn

**Tools Used:** Machine Learning - Python 


## 📑 Table of Contents

1. 🌱 [Data Preprocessing](#data-preprocessing)
2. 🔍 [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. 📊 [Train & Apply Churn Prediction Model](#train-apply-churn-prediction-model)
4. 💡 [Key Findings and Recommendations for Retention](#key-findings-and-recommendations-for-retention)
5. 🤖 [Create A Model For Predicting Churn](#create-a-model-for-predicting-churn)
6. 🧑‍💻 [Customer Segmentation Using Clustering](#customer-segmentation-using-clustering)


## 📌 Background & Overview

### **🎯 Objective**

This project focuses on **predicting and segmenting churned users** in an e-commerce business to **develop effective retention strategies**. By leveraging **Machine Learning & Python**, this project aims to:

✔️ Identify key behaviors and patterns of churned users.

✔️ Develop a predictive model to forecast customer churn.

✔️ Segment churned users to personalize retention offers and promotions.


**❓ What Business Question Will It Solve?**

✔️ What factors contribute to customer churn in e-commerce?

✔️ How can we predict churned users and take proactive measures?

✔️ How to build an accurate churn prediction model?

✔️ How can we segment churned users for targeted promotions?


**👤 Who Is This Project For?**

✔️ Data Analysts & Business Analysts – To gain insights into churn behavior and retention strategies.

✔️ Marketing & Customer Retention Teams – To design data-driven promotional campaigns.

✔️ Decision-makers & Stakeholders – To reduce churn and improve customer lifetime value.


## 📂 **Dataset Description & Data Structure**

### 📌 **Data Source**  
**Source:** The dataset is obtained from the e-commerce company's database.  
**Size:** The dataset contains 5,630 rows and 20 columns.  
**Format:** .xlxs file format.

### 📊 **Data Structure & Relationships**

1️⃣ **Tables Used:**  
The dataset contains only **1 table** with customer and transaction-related data.

2️⃣ **Table Schema & Data Snapshot**  
**Table: Customer Churn Data**

<details>
  <summary>Click to expand the table schema</summary>

| **Column Name**              | **Data Type** | **Description**                                              |
|------------------------------|---------------|--------------------------------------------------------------|
| CustomerID                   | INT           | Unique identifier for each customer                          |
| Churn                        | INT           | Churn flag (1 if customer churned, 0 if active)              |
| Tenure                       | FLOAT         | Duration of customer's relationship with the company (months)|
| PreferredLoginDevice         | OBJECT        | Device used for login (e.g., Mobile, Desktop)                 |
| CityTier                     | INT           | City tier (1: Tier 1, 2: Tier 2, 3: Tier 3)                   |
| WarehouseToHome              | FLOAT         | Distance between warehouse and customer's home (km)         |
| PreferredPaymentMode         | OBJECT        | Payment method preferred by customer (e.g., Credit Card)     |
| Gender                       | OBJECT        | Gender of the customer (e.g., Male, Female)                  |
| HourSpendOnApp               | FLOAT         | Hours spent on app or website in the past month              |
| NumberOfDeviceRegistered     | INT           | Number of devices registered under the customer's account   |
| PreferedOrderCat             | OBJECT        | Preferred order category for the customer (e.g., Electronics)|
| SatisfactionScore            | INT           | Satisfaction rating given by the customer                    |
| MaritalStatus                | OBJECT        | Marital status of the customer (e.g., Single, Married)       |
| NumberOfAddress              | INT           | Number of addresses registered by the customer               |
| Complain                     | INT           | Indicator if the customer made a complaint (1 = Yes)         |
| OrderAmountHikeFromLastYear  | FLOAT         | Percentage increase in order amount compared to last year   |
| CouponUsed                   | FLOAT         | Number of coupons used by the customer last month            |
| OrderCount                   | FLOAT         | Number of orders placed by the customer last month           |
| DaySinceLastOrder            | FLOAT         | Days since the last order was placed by the customer        |
| CashbackAmount               | FLOAT         | Average cashback received by the customer in the past month  |

</details>

## **⚒️ Main Process

## 1️⃣ **Data Preprocessing**  

📌 Import Necessary Libraries

[In 1]: 

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score, 
    precision_score, f1_score, confusion_matrix, silhouette_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
```
📂 Mount Google Drive to Access Files

[In 2,3]: 

```python
# Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_excel("/content/drive/MyDrive/K42_Lê Anh Tuấn _ Statistic & ML/churn_prediction.xlsx")
print(df.shape)
df.head(5)
```


[Out 3]:

![Image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/fd628395ff011ba158a382848dd5d9b8a3a5083c/documents/p1.png)

#### **💡 Data Understanding**

📌 Before performing any analysis or modeling, I carried out several steps to preprocess the data:

**📝 Checked Dataset Structure**  

After checking the general structure of the dataset, this gave me an overview of the number of rows, columns, and data types for each feature, along with summary statistics.

  - The dataset contains 5,630 rows and 20 columns, with a mix of numeric and categorical variables.
  - Missing values were identified in several columns, such as `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, etc.

**📝 Checked for Missing Values**  
Missing values were detected in multiple columns. The columns with missing values are:

   - `Tenure` - 264 missing values
   - `WarehouseToHome` - 251 missing values
   - `HourSpendOnApp` - 255 missing values
   - `OrderAmountHikeFromlastYear` - 265 missing values
   - `CouponUsed` - 256 missing values
   - `OrderCount` - 258 missing values
   - `DaySinceLastOrder` - 307 missing values

**📝 Checked for Duplicates**  

Aftering checkeing for duplicate rows in the dataset and found that there were no duplicate entries.

### **💡 Summary** 

- The dataset contained missing values in several columns. The missing values were **handled by replacing them with the mean**, which prepared the data for further analysis and modeling.
- The dataset contains words with the **same meaning** but written differently. These should be **standardized into a single form**.

**📝 Missing Value Handling**  

[In 4]:

```python
# Define the list of columns with missing values
cols_missing = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']

# Replace missing columns with median
for col in cols_missing:
    # Fill missing values in each column with the median of that column
    df[col].fillna(value= df[col].median(), inplace=True)
```

**📝 Merging Columns with Similar Data but Different Names**  

[In 5]:

```python
# Merge columns with the same data but different names
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
    'COD': 'Cash on Delivery',  # Replace 'COD' with 'Cash on Delivery'
    'CC': 'Credit Card'  # Replace 'CC' with 'Credit Card'
})
```

## 2️⃣ **Exploratory Data Analysis (EDA)**

**📝 Explored Continuous Variables**  

Explored the distribution of continuous variables to better understand their uniqueness and spread. Most of the continuous variables had a limited number of unique values, but this is reasonable given the context of the dataset.

**📝 Univariate Analysis**  
- Categorical variables like `PreferredLoginDevice`, `Gender`, `MaritalStatus`, and others were analyzed through count plots, which gave insights into the distribution of each category.
  
- Continuous variables like `Tenure`, `SatisfactionScore`, `CashbackAmount`, and others were examined using boxplots. It was observed that the outliers in these columns. However, **outliers** reasonable and **should be kept** because they represent **distinguishing characteristics for predicting churn**.


**📝 Final Data Inspection**

[In 6]:
```python
# Print dataset info to verify missing values and column types
print(df.info())
```

[Out 6]:

![Image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/fd628395ff011ba158a382848dd5d9b8a3a5083c/documents/p2.png)

## 3️⃣ **Train & Apply Churn Prediction Model**

### **📝 Encoding**

After preprocessing the dataset, encoding was applied to the categorical features:

1. **One-Hot Encoding**:
   - The columns with categorical features having a small number of unique values were encoded using **one-hot encoding**. This creates binary columns for each unique value, making it easier for the model to process categorical data.
   - The columns encoded are:
     - `PreferredLoginDevice`
     - `PreferredPaymentMode`
     - `PreferedOrderCat`
     - `MaritalStatus`

2. **Label Encoding**:
   - The `Gender` column was encoded using **label encoding** to convert categorical labels into numerical values (0 or 1).
   
3. **Dropped Unnecessary Column**:
   - The `CustomerID` column was dropped since it is a unique identifier and does not contribute to the prediction model.

[In 7]:

```python
# Apply one-hot encoding to categorical columns with a small number of unique values
df_encoded = pd.get_dummies(df, columns=['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice'])

# Apply label encoding to the 'Gender' column (converts categorical labels to numerical values)
label_encoder = LabelEncoder()
df_encoded['Gender'] = label_encoder.fit_transform(df_encoded['Gender'])

# Drop the 'CustomerID'
df_encoded = df_encoded.drop(columns=['CustomerID'])
```

[Out 7]:

![Image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/fd628395ff011ba158a382848dd5d9b8a3a5083c/documents/p3.png)

### **📝 Split Data into Features (X) and Target (y)**

[In 8]:

- The dataset was split into **features (X)** and **target (y)**, where:
  - **X** contains all the independent variables (features), and
  - **y** contains the target variable `Churn`.

```python
# Split the data into features (X) and target (y)
x = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']  # Target

# Split into training and testing sets (70/30 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

[In 9]:

###📝 **Standardize the Features Using StandardScaler**
- The features were standardized using **StandardScaler** to ensure that all features have a mean of 0 and a standard deviation of 1. This step is important because many machine learning algorithms perform better when the features are on the same scale.
- The **training set** was fitted and transformed, while the **test set** was only transformed (to avoid data leakage).

```python
# Standardize the features using StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

### 📝 **Apply Model - Random Forest Classifier**

- **Random Forest Classifier** was trained on the scaled features.
- The model was evaluated using accuracy on both the **training** and **test** sets.

```python
# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train_scaled, y_train)

# Make predictions on training and test sets
y_pred_train = clf.predict(x_train_scaled)
y_pred_test = clf.predict(x_test_scaled)

# Evaluate model accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)

# Print the results
print(f'Training Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Training Balanced Accuracy: {train_balanced_acc:.4f}')
print(f'Test Balanced Accuracy: {test_balanced_acc:.4f}')
```
### **💡Conclusion: ** 

The evaluation of the model's performance on both the training and test sets is as follows:

- **Training Accuracy:** 1.0000  
  - The model achieves perfect accuracy on the training set. However, this could indicate overfitting, as the model has learned the training data very well.
  
- **Test Accuracy:** 0.9343  
  - The model performs well on the test set, with a high accuracy of 93.43%. This suggests that the model generalizes well to unseen data, although there is a slight drop from the training accuracy, which is expected.

- **Training Balanced Accuracy:** 1.0000  
  - The training balanced accuracy is perfect (1.0000), indicating that the model is equally good at predicting both classes (Churn and Non-Churn) in the training set.

- **Test Balanced Accuracy:** 0.8421  
  - The balanced accuracy on the test set is 84.21%, which is still quite good. This metric takes into account both classes and gives a better sense of the model's performance in imbalanced datasets.

-> The model performs well overall, with **high accuracy** and **balanced accuracy** on both the training and test sets. The **perfect training accuracy** indicates a **risk of overfitting**, but the **test accuracy** and **balanced accuracy** show that the model still **generalizes well to new data**. It is essential to **monitor the model on additional datasets** to confirm its **robustness in real-world scenarios**.

### 📝 **Apply Random Forest To Find Important Features**

[In 10]:

```python
# Get feature importances from the best model
feats = {feature: importance for feature, importance in zip(x_train.columns, best_clf.feature_importances_)}

# Create a DataFrame for feature importances
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importances'})
importances = importances.sort_values(by='Importances', ascending=True).reset_index()

# Plot the feature importances
plt.figure(figsize=(10,10))
plt.barh(importances.tail(20)['index'], importances.tail(20)['Importances'])  # Plot top 20 features
plt.title('Feature Importance')
plt.show()
```

[Out 10]:

![image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/fd628395ff011ba158a382848dd5d9b8a3a5083c/documents/p4.png)

### **💡Conclusion: ** 

Base on the image, we can conclude that top 5 important features hwich directly affecting to churn behaviors are: Tenture, CashbackAmount, WarehouseTohome, Complain, DaySinceLastOrder.
These features play a crucial role in predicting whether a customer is likely to churn. Tenure and Days Since Last Order indicate customer engagement, while Cashback Amount and Complaints reflect satisfaction levels. Warehouse to Home Distance may influence delivery experience, impacting customer retention. 
Then, we will plot a histogram chart to visualize the differences between churn and non-churn behavior for the top important features. This will help us identify patterns and understand how these features contribute to customer churn. 

### 📝 **Plot Histogram**

![Image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/fd628395ff011ba158a382848dd5d9b8a3a5083c/documents/p5.png)

## 4️⃣ **Key Findings and Recommendations for Retention**  

### **💡Findings:**
| Metric                     | Churn (Blue) | Non-Churn (Yellow) | Insight | Recommendation |
|----------------------------|-------------|---------------------|---------|----------------|
| **Tenure (Customer Lifespan)** | 80% leave within 5 months, very few stay beyond 10 months | More evenly distributed, many stay over 20 months | **Churned customers leave early**, making the initial experience crucial. | Strengthen onboarding with welcome programs & early engagement incentives. |
| **CashbackAmount (Cashback Received)** | Average around 100-200, widely distributed | Mostly concentrated between 120-250 | **Churned customers receive less cashback**, lowering perceived value. | Introduce tiered cashback, bonuses, or alternative rewards. |
| **WarehouseToHome (Delivery Time)** | Wide distribution, avg. 15-30 days, some over 35 days | Mostly under 20 days, rarely over 25 days | **Longer delivery times increase churn.** | Optimize logistics, reduce shipping time, and offer real-time tracking. |
| **Complain (Customer Complaints)** | 50% complain, 50% leave without feedback | Most don’t complain, only 10-15% do | **Churned customers complain more or leave silently.** | Use AI chatbots, proactive outreach, and offer compensations. |
| **DaySinceLastOrder (Days Since Last Order)** | Mostly over 10 days, many exceed 20 | Mostly under 10 days, rarely over 15 | **Churned customers order less frequently.** | Launch re-engagement discounts, personalized reminders, and subscription models. |

## 5️⃣ **Create A Model For Predicting Churn** 

 The model will be trained using the top 5 features impacting churn behavior: Tenure, CashbackAmount, WarehouseToHome, Complain, and DaySinceLastOrder.
 
 [In 10]:

```python
import time
from sklearn.metrics import recall_score

# Lấy 5 biến quan trọng nhất
top_features = ['Tenure', 'CashbackAmount', 'WarehouseToHome', 'Complain', 'DaySinceLastOrder']
x_q2 = df_encoded[top_features]
y_q2 = df_encoded['Churn']

# Chia tập dữ liệu (70% train, 15% validation, 15% test)
x_train_q2, x_temp, y_train_q2, y_temp = train_test_split(x_q2, y_q2, test_size=0.3, random_state=42)
x_val_q2, x_test_q2, y_val_q2, y_test_q2 = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Chuẩn hóa
scaler_q2 = StandardScaler()
x_train_q2_scaled = scaler_q2.fit_transform(x_train_q2)
x_val_q2_scaled = scaler_q2.transform(x_val_q2)
x_test_q2_scaled = scaler_q2.transform(x_test_q2)

print(f"Kích thước tập Train: {x_train_q2.shape}")
print(f"Kích thước tập Val: {x_val_q2.shape}")
print(f"Kích thước tập Test: {x_test_q2.shape}")

```

### 📝 **Choose Model**

**1. Choose metric for evaluating**

- The primary goal is to **accurately identify customers likely to churn** and implement retention strategies.  
- If the model **misses too many churned customers** (high **False Negatives - FN**), the business will lose customers that could have been retained.  
- **Losing customers (FN) causes more damage** than mistakenly targeting non-churned customers (FP) for retention efforts → **Recall is prioritized.**

**2. Model Comparison & Selection**  

[In 11] :

```python
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    model.fit(x_train_q2_scaled, y_train_q2)
    y_val_pred = model.predict(x_val_q2_scaled)
    recall = recall_score(y_val_q2, y_val_pred)
    results[name] = recall
    print(f"{name} - Recall trên tập Validation: {recall:.4f}")

# Vẽ biểu đồ so sánh
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title('So sánh Recall Score giữa các mô hình (Validation Set)')
plt.ylabel('Recall Score')
plt.show()
```

[Out 12] :

![image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/3f45b4a9fbe0d59515ca9d9d24b12ffdb5e43e48/documents/p6.png)

- After testing five models, **Random Forest achieved the highest Recall score**.  
- → **Select Random Forest and proceed with fine-tuning** to improve performance.  

**3. Apply Model & Fine tune**


[In 12]: 

Fine-tune XGBoost Model 

```python
# Lưới tham số cho Random Forest
param_dist_rf = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Tối ưu hóa mô hình dựa trên Recall score
rf_random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist_rf,
    n_iter=20,
    cv=5,
    scoring='recall',
    random_state=42,
    n_jobs=-1
)

print("Đang tối ưu hóa mô hình Random Forest (Question 2)...")
rf_random_search.fit(x_train_q2_scaled, y_train_q2)

# Lấy mô hình tốt nhất
best_rf_model = rf_random_search.best_estimator_
print(f"Best Parameters: {rf_random_search.best_params_}")
print(f"Best Validation Recall: {rf_random_search.best_score_:.4f}")
```

[Out 12]:  

![Image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/10679a98c1cdfb94073397e17bab13bed967c7d4/documents/Screenshot%202026-07-01%20170032.png)

## 6️⃣ **Customer Segmentation Using Clustering**  

### 📝 **PCA**

[In 13]

```python
# One-hot encoding
df_churned_encoded = pd.get_dummies(df_churned,
                                    columns=['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus'])
# Label encoding
le = LabelEncoder()
df_churned_encoded.loc[:, 'Gender'] = le.fit_transform(df_churned_encoded['Gender'])

# Normalize data
scaler = RobustScaler()
df_churned_final = scaler.fit_transform(df_churned_encoded)

# Initialize PCA with 90% variance retention
pca = PCA(n_components=0.90)

# Apply PCA transformation to the dataset
pca_final = pca.fit_transform(df_churned_final)

# Convert PCA into dataframe
pca_df = pd.DataFrame(pca_final, columns=[f"col{i+1}" for i in range(pca_final.shape[1])])

# Print the number of principal components retained
print(f'Number of principal components retained: {pca_final.shape[1]}')

# Print the explained variance ratio for each principal component
pca.explained_variance_ratio_

# Print the explained variance ratio for each principal component
pca.explained_variance_ratio_
```

[Out 13]:

![Image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/4051232dc603da8a6c953bde21493fff9585c304/documents/p8.png)

### 📝 **Apply Model & Clustering**

**Step 1: Choosing K**

[In 14]:

```python
#Calculate KMeans
from sklearn.cluster import KMeans
ss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, n_init=10, random_state=42, init='k-means++')
  kmeans.fit(pca_df)
  ss.append(kmeans.inertia_)

#Plot the Elbow
plt.figure(figsize = (6,4))
plt.plot(range(1,11), ss, marker=0, linestyle='--')
plt.title('Elbow Methos')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

[Out 14]:

![image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/4051232dc603da8a6c953bde21493fff9585c304/documents/p9.png)

-> We will choose **K=4**

**Step 2: Apply Model**

```python
# Initialize KMeans with 4 clusters
kmeans = KMeans(n_clusters=4, n_init=10, init='k-means++')

# Apply KMeans and get predicted labels
predicted_labels = kmeans.fit_predict(pca_df)

# Add cluster labels to dataframes
pca_df['cluster'] = predicted_labels
df_churned_encoded['cluster'] = predicted_labels
df_churned['cluster'] = predicted_labels
```

**Step 3: Evaluate Model**

[In 15]:

```python
# Calculate and print silhouette score
sil_score = silhouette_score(pca_df, predicted_labels)
print(sil_score)
```

[Out 15]: 0.16964175320877975

**Step 4: Visulize Distribution & Clusters**

![image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/5708060c2616131c1e78a2a773a08d4dde77f7fc/documents/Screenshot%202026-07-01%20170804.png)

![image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/5708060c2616131c1e78a2a773a08d4dde77f7fc/documents/Screenshot%202026-07-01%20170720.png)


### **💡Conclusion: **
- **PCA does not retain the significant meaning of the data** (the sum of the explained variance ratio is too low).
- When applying the **Elbow method**, no clear elbow points are visible.
- **Our hypothesis** is that the data is **sporadic**, meaning there are no clear patterns between the data points, and therefore, clustering into distinct groups is challenging.
- **Silhouette score is also low**, indicating that the clusters are not well-separated.

### **💡Suggestions:**
- **Use clustering methods that do not require a fixed number of clusters**. We suggest trying a **Hierarchical Clustering model**, which can provide better-defined clusters without the need to predefine the number of clusters.

**Step 5: Apply Dendrogram**

[In 16]:

```python
X_pca = pca_df.values

# Draw dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(X_pca, method='ward'))
plt.axhline(y=6, color='r', linestyle='--')
plt.show()
```

[Out 16]:

![image](https://github.com/LeAnhTuan289/E-Commerce-Customer-Churn-Prediction-Machine-Learning---Python/blob/a559229b26ab6e3a671d093e4e7e41584fcf6321/documents/Screenshot%202026-07-01%20171422.png)

**Step 6: Evaluate Dendrogram**

[In 17]:
```python
# Apply Agglomerative Clustering with 3 clusters
agg_clustering = AgglomerativeClustering(n_clusters=3)
clusters = agg_clustering.fit_predict(pca_df)

# Calculate the Silhouette Score
sil_score = silhouette_score(pca_df, clusters)
print(f"Silhouette Score: {sil_score}")

# Visualize the clusters (if pca_df is 2D or 3D)
plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], c=clusters, cmap='viridis')
plt.title('Agglomerative Clustering Results')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
```

[Out 17]:

Silhouette Score: 0.32238739179773573 -> **low**

## 7️⃣ **Recommendation for Clustering**

- **Gather more data on churned users**:  
  - To improve the model, we can **collect additional data** on churned users, either by gathering real data from the business or by using our **supervised model to predict churn**. The predicted churn data can serve as a reference or a new feature for refining the clustering model.
  
- **Run promotions for churned users**:  
  - **Offer promotions** to all users identified as churned, and **track the results**. These insights can be used as **additional features** in future models, helping to enhance the accuracy and effectiveness of churn prediction over time.

These steps would help **improve data quality** and **increase the predictive power** of the model by incorporating real-world results and feedback.
