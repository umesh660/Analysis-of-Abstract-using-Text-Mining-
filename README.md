#  Analysis of Journal Papers

## Introduction
The dataset `journal_data.csv` contains information on papers published in three journals between 2000 and 2022. This README document outlines the tasks performed on the dataset and provides insights into the analysis.

## Dataset Overview
- **Entries**: 4385 papers
- **Columns**: Title, Journal Name, Year, Pages, Authors, Views, Citations, Altmetric Score, Abstract

## Tasks and Analysis

### 1. Bag of Words and TF-IDF
**Objective:** Explore the popular words in abstracts and their distribution.
**Method:** Preprocess abstracts, generate a corpus, compute TF-IDF scores.
**Findings:** Investigated word distributions across years and journals.

### 2. Topic Modelling
**Objective:** Use LDA to discover topics within the papers.
**Method:** Applied LDA to abstracts to identify themes.
**Insights:** Examined topic distributions over time and across journals.

### 3. Regression
**Objective:** Predict number of citations based on abstract features.
**Method:** Trained multiple regression model using abstract data.
**Results:** Analyzed predictors of citation counts.

### 4. Classification and Association
**Objective:** Classify papers by journal and find word associations.
**Approach:** Built classification models and association rules.
**Observations:** Compared journal-specific word patterns.

### 5. Clustering
**Objective:** Identify natural groupings of papers.
**Method:** Applied clustering algorithms to abstract data.
**Outcomes:** Explored clusters and their top distinguishing terms.

### 6. PCA Visualization
**Objective:** Visualize clusters in 2D space.
**Approach:** Applied PCA to reduce dimensions for visualization.
**Visual Results:** Presented clustered groups in a visual format.

## Hypotheses and Conclusions

### Hypotheses
- Explored hypotheses about journal focus areas and temporal changes.

### Validation
- Supported findings with data-driven evidence.

### Conclusions
- Summarized key insights and patterns discovered.

## Next Steps
- **Further Analysis:** Consider additional factors like author impact, keywords trends, etc.
- **Refinement:** Improve models and analyses for more accurate predictions.
- **Visualization Enhancement:** Enhance visualizations for clearer communication.

## Dependencies
- **Software:** Python, Pandas, Scikit-learn, NLTK, Matplotlib, etc.
- **Data Source:** `journal_data.csv` (downloaded from Blackboard).
