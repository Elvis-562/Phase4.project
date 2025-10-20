## Apple and Google Products Twitter Sentiment Analysis

**Authors:**

Elvis Wanjohi (Team Leader)

Jessica Gichimu

Jesse Ngugi

Stephen Gachingu

Latifa Riziki

## 1. Business Understanding

### 1.1 Business Overview
Apple and Google are global technology companies that offer a variety of electronic products and software services. Like any big company, there is public opinion of the products and services they bring to the market. As leaders in the competitive tech industry, their operations depend heavily on customer satisfaction and public perception of their brands.

Customer sentiment plays a key role for both companies. Negative opinions expressed online can affect brand image and influence purchasing decisions. By analyzing customer feedback from social media platforms such as Twitter, these companies can gain insights into how users feel about their products and services. This can help identify areas for improvement, respond to customer concerns, and strengthen their brand reputation.

### 1.2 The Problem
Apple and Google’s success relies heavily on maintaining strong customer satisfaction and positive public perception. With users actively sharing opinions on Twitter, analyzing this feedback has become important for improving products and strengthening brand reputation.

The main challenge is effectively analyzing this data to understand customer sentiment. To address this, this project builds a sentiment analysis model that classifies tweets about Apple and Google products into positive, negative, or neutral categories.

### 1.3 Project Objectives
#### 1.3.1 Main Objective
The main objective of this project is to develop a sentiment classification model that analyzes tweets about Apple and Google products and classifies them as positive, negative, or neutral.

#### 1.3.2 Specific Objectives
The specific objectives of the project are:

1. Determine the products and services from Apple or Google that have the largest negative, positive, and neutral feedback.

2. Preprocess the data through processes such as Vectorization and tokenization, handling missing values, and creating new features with respect to user behavior.

3. Evaluate the model performance using Precision, Recall, Accuracy Score, ROC, and F1score.

4. Compare different classification models to determine which performs best for this dataset.

#### 1.3.3 Research Questions
To ensure the analysis directly addresses the business problem, the following research questions were defined:

1. Which products and services from Apple or Google have the largest negative, positive, and neutral feedback?

2. Which features influence user behavior?

3. Which classifier model had the best Precision, Recall, Accuracy Score, ROC, and F1 score?

4. Which classification model performs best for this dataset?

Answering these will provide the tech companies with data-driven insights, which will in turn enhance long-term profitability.

### 1.4 Success Criteria
The success of this project will be assessed in the following ways:

1. It should generate insights into how users feel about their products and services.

2. A machine learning model should be successfully developed that automatically determines the sentiment of a tweet based on the words and tone used in the text.

## 2. Data Understanding
This section explains the Twitter sentiment dataset used for the project. The dataset contains tweets about Apple and Google products, forming the basis for building a sentiment classification model.

The aim is to understand the structure and content of the dataset. This involves reviewing the available features, checking their data types, and identifying potential issues such as missing values or inconsistencies.

By exploring the data at this stage, it is possible to detect quality concerns early and begin considering how the dataset can best be prepared for text cleaning, preprocessing, and model development.

## 3. Data Preparation
This section focuses on preparing the dataset for analysis by applying systematic data cleaning, feature engineering procedures, and data pre-processing procedures like normalization.

The aim is to transform raw, unstructured text into a clean and structured format suitable for effective sentiment classification.

These steps ensure data consistency, reduce noise, and enhance the quality of features used for modeling.

The newly created columns from feature engineering include:

- Character

- Words

- Sentences

The steps include:

- Cleaning: Removes URLs, @mentions, and hashtags. It expands contractions, normalizes repeated letters, and strips special characters. In addition, it standardizes punctuation and whitespace.

- Normalization: involves converting the text data into a consistent format by converting all the tweets to lowercase.

- Stopword removal: This involves removing words with no significant meaning.

- Tokenization- This involves breaking the texts into smaller words or phrases that the model can understand.

- Pos tagging- It is short for part of speech and involves assigning each text a grammatical category like noun, verb, and adjective.

- Lemmatization- It reduces words to their base root or form.

## 4. Exploratory Data Analysis
The dataset was explored to understand linguistic and sentiment patterns across tweets. Key analysis included:

- Sentiment by Tweet Destination: Examined how positive, negative, and neutral sentiments are distributed across major brands (Apple, Google, iPhone, iPad). Most tweets were not directed (mainly neutral), while Apple and iPad received the highest engagement with predominantly positive sentiment.

- Tweet Length Distribution by Sentiment: Analyzed tweet length (in characters) across sentiment categories to identify expression patterns. The boxplot comparison helped reveal whether emotional tone correlates with shorter or longer tweets.

- Distribution of Tweet length/Character, Word, and Sentence Counts: Explored the linguistic structure of tweets through counts of characters, words, and sentences. This provided insight into tweet complexity, detected anomalies, and identified opportunities for feature engineering to improve model performance.

- General Tweets Dominate: Most tweets (5,431) are not directed and largely neutral, indicating a high volume of general discussions rather than brand-specific mentions.

- Top Mentioned Brands: Among directed tweets, iPad (792) and Apple (541) received the highest engagement, followed by iPad/iPhone Apps (396) and Google (344).

- Positive Sentiment Prevails: Positive emotions dominate across all directed destinations, with very few negative tweets recorded.

- Apple Leads in Favorability: Apple-related products attract the most engagement and exhibit the most favorable overall sentiment among users.

These analyses helped uncover how users express emotions across brands, assess text quality, and inform preprocessing decisions for better sentiment classification.

## 5. Modeling and Evaluation
Four machine learning models were developed and evaluated for tweet sentiment classification:

1. **Logistic Regression**: A linear model used for binary and multiclass classification.

2. **Naive Bayes**: A probabilistic classifier commonly applied in text and sentiment analysis.

3. **LinearSVC**: A support vector machine variant optimized for high-dimensional text data.

4. **Random Forest**: An ensemble model that combines multiple decision trees for robust predictions.

### 5.1 Binary Classification
The dataset was filtered to include only Positive and Negative sentiments.

- Models were trained using TF-IDF vectorization and evaluated using precision, recall, F1-score, and accuracy.

- LinearSVC achieved the best performance with a validation accuracy of 87.58%, indicating superior generalization for binary sentiment prediction.

### 5.2 Multi-Class Classification
Included Positive, Negative, and Neutral sentiments.

Data was split into training (70%), validation (15%), and testing (15%) sets.

Each model was optimized using GridSearchCV within a TF-IDF in a Model pipeline.

Evaluation metrics included Accuracy, F1-Score, and ROC-AUC on both train and validation sets.

**Key Observations:**

1. LinearSVC again delivered the highest validation accuracy (67.2%), confirming its reliability across binary and multiclass tasks.

2. Random Forest showed signs of overfitting, with the train accuracy being 99.4%, indicating the model memorized the data too well, making it unable to generalize well to unseen data.

3. Logistic Regression and Naive Bayes achieved balanced yet moderate performance.

**Model Insights:**

1. TF-IDF features captured meaningful linguistic patterns in tweets.

2. Simpler linear models generalize better than complex ensemble methods.

**Future improvements could involve:**

1. Fine-tuning hyperparameters using stratified cross-validation.

2. Exploring deep learning models (LSTM, BERT.), which could be an option with a larger dataset. An example is a dataset with many tweets.

3. Expanding the dataset to balance sentiment classes.

## 6. Deployment

The trained model was deployed to Streamlit Cloud to make it interactive and easy to use.
Key Steps:
1. Prepared a requirements.txt file listing all required Python packages.

2. Encountered a deployment error caused by a missing path (../requirements.txt).

3. Fixed the issue by removing the invalid line and redeploying.
  
4. Streamlit automatically installed all required libraries and launched the web app successfully.

5. The model is now live and accessible to users for testing and prediction.

App Link:
https://phase4projec-qkr7ewgse2npajgfsamzq3.streamlit.app/

## 7. Technologies Used

- **Python**: Primary programming language

- **Pandas**: Data manipulation and analysis

- **Sklearn**: For modelling purposes

- **Matplotlib**: Data visualization

- **Jupyter Notebook**: Development environment

- **Git**: Commit and push to the remote repository

- **NLTK**: For Natural Language Processing

- **Plotly**: To create interactive visualizations

- **Joblib and Pickle**: To save the trained model.

## 8. Conclusion
This Twitter sentiment analysis project has provided valuable insights into the challenges and opportunities of automated sentiment classification for Apple and Google products. The analysis encompassed both binary and multi-class classification approaches, revealing important patterns about social media sentiment and model performance. Both the binary and multi-class approaches identified LinearSvc as the best performing model, suggesting that the model can be saved and its analysis used to make new predictions.

## 9. Support
For questions or support, please contact:

1. jessengugi99@gmail.com

2. jessica.gichimu@gmail.com

3. stephenmunene092@gmail.com

4. leeelvis562@gmail.com

5. latifariziki5@gmail.com

