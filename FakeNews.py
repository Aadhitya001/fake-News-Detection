import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset (Can be replaced with a real one)
data = {
    'text': [
        'Donald Trump sends out embarrassing tweet',
        'The moon landing was faked by NASA',
        'NASA launches new satellite to study climate change',
        'COVID-19 vaccines are effective and safe',
        'Bill Gates plans to microchip everyone through vaccines'
    ],
    'label': ['FAKE', 'FAKE', 'REAL', 'REAL', 'FAKE']
}

df = pd.DataFrame(data)

# Split data
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.33, random_state=7)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)

# Model Training
