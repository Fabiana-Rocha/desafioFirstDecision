import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler

nltk.download('stopwords')
class PreProcessor:
    def __init__(self, palavras_chave):
        self.palavras_chave = palavras_chave

    def preprocess_text(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(re.sub(r'[^A-Za-z0-9 ]+', ' ', text.lower()))
        tokens = [word for word in tokens if word not in stop_words]
        text = ' '.join(tokens)
        return text

    def cria_rotulos(self, data, coluna_texto):
        if coluna_texto not in data.columns:
            raise ValueError(f"A coluna '{coluna_texto}' não está presente no DataFrame.")

        data['label'] = data[coluna_texto].apply(
            lambda x: any(keyword.lower() in x.lower() for keyword in self.palavras_chave))

        return data

    def divide_conjunto_dados(self, data, coluna_texto, coluna_rotulo, test_size=0.2, random_state=111):
        if coluna_texto not in data.columns or coluna_rotulo not in data.columns:
            raise ValueError(f"As colunas '{coluna_texto}' ou '{coluna_rotulo}' não estão presentes no DataFrame.")

        X_train, X_test, y_train, y_test = train_test_split(data[coluna_texto], data[coluna_rotulo],
                                                            test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    def vetoriza_texto_tfidf(self, X_train, X_test, max_features=5000):
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        return X_train_vectorized, X_test_vectorized, vectorizer

    def aplica_oversampling(self, X_vectorized, y, random_state=111):
        oversampler = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = oversampler.fit_resample(X_vectorized, y)

        return X_resampled, y_resampled







