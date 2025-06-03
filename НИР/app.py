
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Функция подготовки данных ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('Empress of Ireland.csv')
    # Заполняем пропуски
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    # Новые признаки
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Title'] = le.fit_transform(df['Title'])

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked', 'FamilySize', 'IsAlone', 'Title']
    X = df[features]
    y = df['Survived']

    scaler = StandardScaler()
    X[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(X[['Age', 'Fare', 'FamilySize']])

    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- Функция обучения модели ---
def обучить_модель(X_train, y_train, n_estimators):
    модель = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    модель.fit(X_train, y_train)
    return модель

# --- Основное приложение Streamlit ---
def main():
    st.title("Прогноз выживаемости на Empress of Ireland — Random Forest")
    st.write("""
        Веб-приложение для классификации выживаемости пассажиров Empress of Ireland.
        Используйте ползунок для изменения **количества деревьев (n_estimators)** в модели Random Forest.
    """)

    # Загружаем и готовим данные
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Слайдер для выбора гиперпараметра
    n_estimators = st.slider("Количество деревьев (n_estimators)", min_value=10, max_value=200, value=100, step=10)

    # Обучаем модель с выбранным параметром
    модель = обучить_модель(X_train, y_train, n_estimators)

    # Предсказания и метрики качества
    y_pred = модель.predict(X_test)
    точность = accuracy_score(y_test, y_pred)
    точность_положительных = precision_score(y_test, y_pred)
    полнота = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Результаты модели")
    st.write(f"**Точность (Accuracy):** {точность:.3f}")
    st.write(f"**Точность положительных предсказаний (Precision):** {точность_положительных:.3f}")
    st.write(f"**Полнота (Recall):** {полнота:.3f}")
    st.write(f"**F1-мера:** {f1:.3f}")

    # Важность признаков
    важность = модель.feature_importances_
    признаки = X_train.columns
    важность_серии = pd.Series(важность, index=признаки).sort_values(ascending=False)

    st.subheader("Важность признаков")
    fig, ax = plt.subplots()
    sns.barplot(x=важность_серии.values, y=важность_серии.index, ax=ax)
    ax.set_xlabel("Важность")
    ax.set_ylabel("Признаки")
    st.pyplot(fig)

    # Итоговые выводы
    st.markdown("---")
    st.subheader("Итоговые выводы")
    st.write("""
    - Мы подготовили данные, заполнили пропуски и создали новые информативные признаки:
        - Размер семьи (количество родственников на борту)
        - Индикатор путешествия в одиночку
        - Титул пассажира (например, Мисс, Мадам, Редкий)
    - Привели числовые признаки к единой шкале с помощью стандартизации.
    - Построили модель случайного леса с настраиваемым числом деревьев.
    - По метрикам качества (точность, полнота, F1) оценили эффективность модели.
    - Визуализировали важность признаков — это помогает понять, какие факторы сильнее влияют на выживаемость.
    
    Это значит, что благодаря инженерии признаков и настройке модели мы получили инструмент, 
    который может предсказывать вероятность выживания пассажиров на основе их данных с приемлемой точностью. 
    Возможность изменять гиперпараметры позволяет исследовать, как структура модели влияет на качество предсказаний.
    """)

if __name__ == '__main__':
    main()
