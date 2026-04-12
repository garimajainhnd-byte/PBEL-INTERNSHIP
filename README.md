NAME - Garima Jain
BATCH - 8
[PROJECT - AI BASED EMOTION RECOGNIZE SYSTEM FROM TEXT]
# Presentation
[Download PPT](https://drive.google.com/file/d/1ADUUmg4NVoq5Qw5tYfPkqEHCPXQHfI3W/view?usp=sharing)
## Presentation Recording
[Watch Video](https://drive.google.com/file/d/1GyHMd_sHGeINbizxsLMfbyKlinc5JwHs/view?usp=sharing)
# PROJECT - 1
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("go_emotions_dataset.csv")

print("First 5 rows of dataset:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

df = df.dropna()

df = df[['text','joy','anger','sadness']]

print("\nCleaned Data (first 5 rows):\n")
print(df.head())

show_graph = input("\n want to see the emotion distribution graph? (yes/no): ").strip().lower()
if show_graph == 'yes':
    emotion_count = df[['joy','anger','sadness']].sum()
    print("\nEmotion Counts:\n", emotion_count)
    
    emotion_count.plot(kind='bar')
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.show()
else:
    print("Graph skipped.")

df['text'] = df['text'].str.lower()
df['text_length'] = df['text'].apply(len)
print("\nText with length (first 5 rows):\n")
print(df[['text','text_length']].head())

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df['text']
y = df['joy']  

vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)
```
<img width="1417" height="813" alt="image" src="https://github.com/user-attachments/assets/51269f19-5495-4349-9ed6-7138a112e3fb" />
<img width="790" height="673" alt="image" src="https://github.com/user-attachments/assets/8ead34d0-3efc-4c8f-b892-a9956b6ac831" />
<img width="797" height="636" alt="image" src="https://github.com/user-attachments/assets/e11f3c11-1536-4eef-8734-331a858ee107" />



