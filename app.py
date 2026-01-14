import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ======================
# NLTK setup
# ======================
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ======================
# Clean text
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ======================
# Load dataset
# ======================
data = pd.read_csv("Emotion_dataset.csv")

if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

data = data[['text', 'label']]
data.rename(columns={'label': 'emotion'}, inplace=True)
data.dropna(subset=['text'], inplace=True)

data['clean_text'] = data['text'].apply(clean_text)
data = data[data['clean_text'] != ""]

# ======================
# Vectorizer + Model
# ======================
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])
y = data['emotion']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ======================
# LABEL → EMOTION MAPPING
# (based on dataset)
# ======================
emotion_map = {
    0: "sad",
    1: "happy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# ======================
# Emotion → Tips
# ======================
emotion_tips = {
    "sad": "Try talking to someone you trust or go for a short walk.",
    "happy": "That’s great! Keep doing what makes you feel good.",
    "anger": "Take a pause and breathe deeply before reacting.",
    "fear": "Focus on what you can control and take things step by step.",
    "love": "Express it openly. Positive connections matter.",
    "surprise": "Give yourself time to process things calmly."
}

# ======================
# USER INTERACTION
# ======================
print("=== Emotion Detection System created by Ajay Gupta ===\n")

name = input("Hello my friend , what's your name: ").strip()
print(f"\nHi {name}! 😊")
print(f"Tell me something, {name}:\n")

user_text = input()

cleaned = clean_text(user_text)
vector = vectorizer.transform([cleaned])
predicted_label = model.predict(vector)[0]

emotion = emotion_map.get(predicted_label, "neutral")
tip = emotion_tips.get(emotion, "Take care of yourself and stay mindful.")

print(f"\nIt seems you are feeling {emotion} {name}.")
print(f"Listen to me : {tip}")