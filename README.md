## Apple and Google Products Twitter Sentiment Analysis
Authors:

Elvis Wanjohi (Team Leader)

Jessica Gichimu

Jesse Ngugi

Stephen Gachingu

Latifa Riziki

1. Business Understanding
1.1 Business Overview
Apple and Google are global technology companies which offer a variety of electronic products and software services. Like any big company, there is public opinion of the products and services they bring to the market. As leaders in the competitive tech industry, their operations depend heavily on customer satisfaction and public perception of their brands.

Customer sentiment plays a key role for both companies. Negative opinions expressed online can affect brand image and influence purchasing decisions. By analyzing customer feedback from social media platforms such as Twitter, these companies can gain insights into how users feel about their products and services. This can help identify areas for improvement, respond to customer concerns and strengthen their brand reputation.

1.2 The Problem
Apple and Google’s success rely heavily on maintaining strong customer satisfaction and positive public perception. With users actively sharing opinions on Twitter, analyzing this feedback has become important for improving products and strengthening brand reputation.

The main challenge is effectively analyzing this data to understand customer sentiment. To address this, this project builds a sentiment analysis model that classifies tweets about Apple and Google products into positive, negative or neutral categories.

1.3 Project Objectives
1.3.1 Main Objective
The main objective of this project is to develop a sentiment classification model that analyzes tweets about Apple and Google products and classifies them as positive, negative or neutral.

1.3.2 Specific Objectives
The specific objectives of the project are:

Determine the products and services from Apple or Google that have the largest negative, positive and neutral feedback.

Preprocess the data through processes such as; Vectorization and tokenization, handling missing values and creating new features with respect to user behavior.

Evaluate the model performance using Precison, Recall, accuracy score, ROC and F1score.

Compare different classification models to determine which performs best for this dataset.

1.3.3 Research Questions
Which products and services from Apple or Google have the largest negative, positive and neutral feedback?

Which features influence user behavior?

Which classifier model had the best Precison, Recall, accuracy score, ROC and F1 score

Which classification model performs best for this dataset?

1.4 Success Criteria
The success of this project will be assessed in the following ways:

It should generate insights into how users feel about their products and services.

A machine learning model should be successfully developed that automatically determines the sentiment of a tweet based on words and tone used in the text.

2. Data Understanding
This section explains the Twitter sentiment dataset used for the project. The dataset contains tweets about Apple and Google products, forming the basis for building a sentiment classification model.

The aim is to understand the structure and content of the dataset. This involves reviewing the available features, checking their data types and identifying potential issues such as missing values or inconsistencies.

By exploring the data at this stage, it is possible to detect quality concerns early and begin considering how the dataset can best be prepared for text cleaning, preprocessing and model development.

3. Data Preparation
This section focuses on preparing the dataset for analysis by applying systematic data cleaning, feature engineering procedures, data pre-processing procedures like normalization.

The aim is to transform raw, unstructured text into a clean and structured format suitable for effective sentiment classification.

These steps ensure data consistency, reduce noise and enhance the quality of features used for modeling.

The newly created columns from feature engineering include:

Character
Words
Sentences
The steps include:

Cleaning: Removes URLs, @mentions and hashtags. It expands contractions, normalizes repeated letters and strips special characters. In addition, it standardizes punctuation and whitespace.

Normalization: involves converting the text data into a consistent format by converting all the tweets to lowercase.

Stopword removal: This involves removing words with no significant meaning.

Tokenization- This involves breaking the texts into smaller words or phrases that the model can understand.

Pos tagging- It is short for part of speech and involves assigning each text a grammatical category like noun, verb and adjective.

Lemmatization- It reduces words to their base root or form.

4. Exploratory Data Analysis
The dataset was explored to understand linguistic and sentiment patterns across tweets. Key analysis included:

Sentiment by Tweet Destination: Examined how positive, negative, and neutral sentiments are distributed across major brands (Apple, Google, iPhone, iPad). Most tweets were not directed (mainly neutral), while Apple and iPad received the highest engagement with predominantly positive sentiment.

Tweet Length Distribution by Sentiment: Analyzed tweet length (in characters) across sentiment categories to identify expression patterns. The boxplot comparison helped reveal whether emotional tone correlates with shorter or longer tweets.

Distribution of Tweet length/Character, Word, and Sentence Counts: Explored the linguistic structure of tweets through counts of characters, words, and sentences. This provided insight into tweet complexity, detected anomalies, and identified opportunities for feature engineering to improve model performance.

General Tweets Dominate: Most tweets (5,431) are not directed and largely neutral, indicating a high volume of general discussions rather than brand-specific mentions.

Top Mentioned Brands: Among directed tweets, iPad (792) and Apple (541) received the highest engagement, followed by iPad/iPhone Apps (396) and Google (344).

Positive Sentiment Prevails: Positive emotions dominate across all directed destinations, with very few negative tweets recorded.

Apple Leads in Favorability: Apple-related products attract the most engagement and exhibit the most favorable overall sentiment among users.

These analyses helped uncover how users express emotions across brands, assess text quality, and inform preprocessing decisions for better sentiment classification.
5. Modelling and Evaluation
Four machine learning models were developed and evaluated for tweet sentiment classification:

Logistic Regression – A linear model used for binary and multiclass classification.

Naive Bayes – A probabilistic classifier commonly applied in text and sentiment analysis.

LinearSVC – A support vector machine variant optimized for high-dimensional text data.

Random Forest – An ensemble model that combines multiple decision trees for robust predictions.

5.1 Binary Classification
The dataset was filtered to include only Positive and Negative sentiments.

SMOTE was applied to address class imbalance (Positive: 2,928 to 2,049, Negative: 560 to 2,049).

Models were trained using TF-IDF vectorization and evaluated using precision, recall, F1-score, and accuracy.

LinearSVC achieved the best performance with 86.44% accuracy, indicating superior generalization for binary sentiment prediction.

5.2 Multi-Class Classification
Included Positive, Negative, and Neutral sentiments.

Data was split into training (70%), validation (15%), and testing (15%) sets.

Each model was optimized using GridSearchCV within a TF-IDF in a Model pipeline.

Evaluation metrics included Accuracy, F1-Score, and ROC-AUC on both train and validation sets.

Key Observations:

LinearSVC again delivered the highest validation accuracy (67.2%), confirming its reliability across binary and multiclass tasks.

Random Forest showed signs of overfitting with the train accuaracy being (99.4%), indicating the model memorized the data too well making it unable to generalize well to unseen data.

Logistic Regression and Naive Bayes achieved balanced yet moderate performance.

Model Insights

TF-IDF features captured meaningful linguistic patterns in tweets.

Simpler linear models generalized better than complex ensemble methods.

Future improvements could involve:

Fine-tuning hyperparameters using stratified cross-validation.

Exploring deep learning models (LSTM, BERT.) could be an option with a larger dataset eg a dataset with many tweets.

Expanding the dataset to balance sentiment classes.

6. Technologies Used
Python: Primary programming language
Pandas: Data manipulation and analysis
Sklearn: For modelling purposes
Matplotlib: Data visualization
Jupyter Notebook: Development environment
Git: Commit and push to remote repository
NLTK: For Natural Language Processing
Plotly: To create interactive visualizations.
Joblib and Pickle: To save the trained model.
7. Conclusion
This Twitter sentiment analysis project has provided valuable insights into the challenges and opportunities of automated sentiment classification for Apple and Google products. The analysis encompassed both binary and multi-class classification approaches, revealing important patterns about social media sentiment and model performance. Both the binary and multi-class appproaches identified LinearSvc as the best performing model, suggesting that we can save the model and use its analysis to make new predictionn.

8. Support
For questions or support, please contact:

1. jessengugi99@gmail.com
2. jessica.gichimu@gmail.com
3. stephenmunene092@gmail.com
4. leeelvis562@gmail.com
5. latifariziki5@gmail.com

