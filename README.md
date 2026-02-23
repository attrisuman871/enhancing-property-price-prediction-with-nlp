# Enhancing Property Price Prediction Using NLP

## Project Overview
Traditional house price models rely mostly on structured attributes (beds, baths, sqft, location). This project demonstrates how **property listing descriptions** add valuable qualitative signals (e.g., “renovated”, “luxury”, “walkable”, “motivated seller”) that can improve price prediction.

This project follows an end-to-end pipeline combining structured features with text-based NLP features.  
(Presentation included in this repo.)

---

## Objectives
- Extract informative signals from listing descriptions
- Convert text into numeric features using **TF-IDF**
- Explore NLP techniques such as **NER** and **sentiment analysis**
- Compare models using structured-only vs structured + unstructured features
- Evaluate business value of text-driven pricing improvements

---

## Data Pipeline
1. **Structured preprocessing** (numerical + categorical features)
2. **Text cleaning** (lowercasing, tokenization, normalization)
3. **TF-IDF vectorization**
4. **NER exploration** (to detect locations, amenities, entities)
5. **Sentiment analysis (VADER)** tested but excluded due to bias
6. Model training + evaluation

---

## NLP Techniques Used

### TF-IDF Vectorization
TF-IDF was used to represent listing descriptions as weighted term features.

<img width="1150" height="428" alt="image" src="https://github.com/user-attachments/assets/7a276314-7e01-4eb9-8cab-acb11918865a" />


### Named Entity Recognition (NER)
NER was explored to detect entities such as locations, amenities, and property attributes.

<img width="1265" height="784" alt="image" src="https://github.com/user-attachments/assets/cf958c21-3735-4ab7-ac13-0dd053ba7ad6" />

### Sentiment Analysis (VADER)
VADER sentiment scoring was evaluated, but sentiment features were excluded from the final models due to **significant bias**.

<img width="571" height="398" alt="image" src="https://github.com/user-attachments/assets/475ffba1-f6ba-4d52-8e65-e0ea791e1d39" />



## 🤖 Modeling Strategy
- Baseline model Random Forest trained using **structured features only**
- Another Random forest model trained using **structured + TF-IDF features**
- Enhanced model XGBoost trained using **structured + TF-IDF features**
- Models evaluated using standard regression metrics (RMSE / MAE / R²)

---

## 📊 Results

| Model | Features | Key Metric |
|------|----------|-----------|
| Baseline Random Forest | Structured only | R² Score: 0.53 |
|Baseline Random Forest | Structured + TF-IDF| R² Score: 0.68 |
| Enhanced XGBoost | Structured + TF-IDF | R² Score: 0.75 |


---

## Key Insights
- Listing descriptions contain pricing signals not captured in structured fields
- TF-IDF features improve performance over structured-only baseline
- Some NLP features (sentiment) may introduce bias and must be validated carefully

---

## Business Impact
- More accurate pricing improves seller confidence and buyer transparency
- Text-derived features can help detect premium amenities and renovation signals
- Supports better automated valuation models (AVMs)

---

## Future Work
- Add geospatial + neighborhood-level features
- Model explainability (SHAP) for most influential terms/features

---

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, NLTK, TF-IDF, SpaCy 

---

## 👤 Authors
Suman Attri, Aisha Mohammad
