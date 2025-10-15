# sentiment-analysis-codec
# 🧠 Sentiment Analysis using NLP and Logistic Regression


## 🏢 Company Information
**Company Name:** Codec technologies  
**Internship Duration:** 14/09/2025 – 14/11/2025  
**Intern Name:** Usha Rahul  

---

## 🎯 Objective
- To perform sentiment analysis on social media posts.
- To preprocess and clean text data.
- To train a **Logistic Regression model** using **TF-IDF vectorization**.
- To evaluate model performance using accuracy, precision, recall, and F1-score.

---

## 🗂️ Dataset Description
- **Dataset Source:** Kaggle  
- **Dataset Columns:**  
  - `Year`, `Month`, `Day`, `Time of Tweet`, `Text`, `Sentiment`, `Platform`
- **Target Column:** `Sentiment` (Positive, Negative, Neutral)

---

## ⚙️ Steps Involved

### 1️⃣ Importing Libraries
- Import necessary libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and NLTK.

### 2️⃣ Data Loading
- Load the dataset using `pd.read_csv()`.
- Display the first few records using `df.head()`.

### 3️⃣ Exploratory Data Analysis (EDA)
- Check for missing values.
- Analyze sentiment distribution using bar plots.
- Visualize word frequency or review length using histograms.
- Detect outliers using box plots.

### 4️⃣ Data Cleaning and Preprocessing
- Remove URLs, mentions, hashtags, punctuation, and stopwords.
- Convert text to lowercase.
- Tokenize and clean using NLTK stopwords.

### 5️⃣ Text Vectorization
- Use **TF-IDF Vectorizer** to convert cleaned text into numerical features.

### 6️⃣ Splitting Dataset
- Split the data into training and testing sets using `train_test_split()`.

### 7️⃣ Model Building
- Train a **Logistic Regression model**.
- Use class weights to handle imbalance if needed.

### 8️⃣ Model Evaluation
- Evaluate performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Display the classification report and confusion matrix.

### 9️⃣ Results
- Accuracy obtained: **68%**
- Positive class performed best, Neutral class moderately, Negative class needs improvement.

## 📊 Results Summary
| Metric | Score |
|--------|--------|
| Accuracy | 0.68 |
| Best Class | Positive |
| Improvement Needed | Negative Sentiment Detection |

---

## 💡 Conclusion
- The model performs well for **positive and neutral sentiments**.
- The project demonstrates the ability to apply **NLP preprocessing and logistic regression** for sentiment classification.
- Future improvements: Use advanced models (SVM, Random Forest, LSTM, BERT).

---

## 🧰 Tools and Technologies Used
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn  
- **Techniques:** TF-IDF Vectorization, Logistic Regression, Text Preprocessing  
- **Platform:** Jupyter Notebook  

---



## 🏁 Final Note
> “Data is powerful when turned into insight — and NLP makes that insight meaningful.”
