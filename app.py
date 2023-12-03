import re
import pandas as pd
import sqlite3
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

class DatabaseClass:
    def __init__(self):
        # Ініціалізація  об'єкта бази даних
        self._cursor = sqlite3.connect('database.db').cursor()

    def select_word_avg_freq(self):
        #виведення для кількості тегів частин мови у pos_frequencies
        query_total_pos_tags = '''
            SELECT COUNT(pos_tag) AS total_pos_tags
            FROM pos_frequencies;
        '''
        result_total_pos_tags = self._cursor.execute(query_total_pos_tags)
        print("\nTotal POS Tags:")
        print(result_total_pos_tags.fetchone()[0])

        #виведення для кількості лем у lemma_frequencies
        query_total_lemmas = '''
            SELECT COUNT(lemma) AS total_lemmas
            FROM lemma_frequencies;
        '''
        result_total_lemmas = self._cursor.execute(query_total_lemmas)
        print("\nTotal Lemmas:")
        print(result_total_lemmas.fetchone()[0])

        #виведення для кількості унікальних слів у word_frequencies
        query_unique_words = '''
            SELECT COUNT(DISTINCT word) AS unique_words
            FROM word_frequencies;
        '''
        result_unique_words = self._cursor.execute(query_unique_words)
        print("\nUnique Words:")
        print(result_unique_words.fetchone()[0])

        #виведення для топ-10 слів за кількістю вживань
        query_top_words = '''
            SELECT
                word,
                SUM(freq_1) AS total_usage
            FROM
                word_frequencies
            GROUP BY
                word
            ORDER BY
                total_usage DESC
            LIMIT 10;
        '''
        result_top_words = self._cursor.execute(query_top_words)
        print("\nTop 10 Words:")
        for row in result_top_words:
            print(row)

        #виведення для топ-10 лем за кількістю вживань
        query_top_lemmas = '''
            SELECT
                lemma,
                SUM(freq_1) AS total_usage
            FROM
                lemma_frequencies
            GROUP BY
                lemma
            ORDER BY
                total_usage DESC
            LIMIT 10;
        '''
        result_top_lemmas = self._cursor.execute(query_top_lemmas)
        print("\nTop 10 Lemmas:")
        for row in result_top_lemmas:
            print(row)


# Функція для перевірки, чи написано слово латиницею
def is_latin(word):
    return bool(re.match(r'^[a-zA-Z]+$', word))

# Функція для виведення слів кожної частини мови у консоль
def print_words_by_pos(pos_tags, words):
    pos_dict = {}
    for pos, word in zip(pos_tags, words):
        if pos is not None:
            pos = pos.capitalize()
            if pos not in pos_dict:
                pos_dict[pos] = []
            pos_dict[pos].append(word)

    print("Слова за частинами мови:")
    for pos, words_list in pos_dict.items():
        print(f"{pos}: {', '.join(words_list)}")

# Просимо користувача ввести назву файлу, який хочемо обробити і відкриваємо його
filename = input("Введіть назву файлу (разом з розширенням), який ви хочете обробити: ")
with open(filename, encoding="utf-8") as data_1:
    text1 = data_1.read().lower()

    # Вилучаємо цифри з тексту, замінючи на пустий рядок
    text1 = re.sub(r'\d', '', text1)
    # Використовуємо регулярний вираз для токенізації
    splitted_1 = re.findall(r'\b\w+(?:[-\'’]\w+)*\b', text1)

    # Видаляємо токени, що містять лише латинські літери
    non_latin_words = [word for word in splitted_1 if not is_latin(word)]

    # Обмежуємо кількість токенів до 20000
    non_latin_words = non_latin_words[:20000]

    # Виводимо токени та їхню кіл-сть у консоль
    print("Виділені словоформи:")
    print(", ".join(non_latin_words))
    token_count = len(non_latin_words)
    print(f"\nКількість словоформ: {token_count}")

    # Створюємо DataFrame для латинських слів
    latin_words = [word for word in splitted_1 if is_latin(word)]
    latin_df = pd.DataFrame(latin_words, columns=['word'])

    # З'єднуємося з базою даних SQLite
    conn = sqlite3.connect('database.db')

    # Записуємо DataFrame в базу даних
    latin_df.to_sql('latin_words', conn, if_exists='replace', index=False)

    # Обновлюємо основний текст, видаляючи латинські слова
    text1 = ' '.join(non_latin_words)

    # Розбиваємо на 20 груп по 1000 слів
    sample_size = 1000
    num_samples = len(non_latin_words) // sample_size
    token_groups = []  # створюємо список для зберігання груп токенів

    words_count = 0
    current_sample = []  # створюємо список для зберігання поточної групи

    # Додаємо токени до поточної групи
    for token in non_latin_words:
        current_sample.append(token)
        words_count += 1

        if words_count == sample_size:
            token_groups.append(current_sample)
            current_sample = []
            words_count = 0

    # Обробка залишкових токенів
    if current_sample:
        token_groups.append(current_sample)

    # Створюємо частотний словник для кожної вибірки
    frequency_dicts = []

    for i, group in enumerate(token_groups):
        frequency_dict = {}
        for word in group:
            frequency_dict[word] = frequency_dict.get(word, 0) + 1
        frequency_dicts.append(frequency_dict)

    # Створюємо DataFrame для слововживань
    df = pd.DataFrame(frequency_dicts).fillna(0).astype(int).T
    df.columns = [f'freq_{i + 1}' for i in range(num_samples)]

    # Додаємо колонку 'gen_freq' та рахуємо загальну частоту для кожного слова
    df.insert(0, 'gen_freq', df.sum(axis=1))

    # Замінюємо індекси на 'word'
    df.index.name = 'word'
    df.reset_index(inplace=True)
    df.index += 1

    # Записуємо DataFrame в базу даних
    df.to_sql('word_frequencies', conn, if_exists='replace', index=False)

    # Морфологічний аналізатор для української мови
    morph = MorphAnalyzer(lang='uk')

    # Функція для отримання леми слова
    def get_lemma(word):
        parsed_word = morph.parse(word)[0]
        return parsed_word.normal_form

    # Застосовуємо функцію до всіх слів у тексті і отримуємо список лем
    lemmas = [get_lemma(word) for word in non_latin_words]

    # Розбиваємо на 20 груп по 1000 лем
    lemma_groups = [lemmas[i * sample_size: (i + 1) * sample_size] for i in range(num_samples)]

    # Створюємо частотний словник для кожної вибірки лем
    lemma_frequency_dicts = []
    for i, group in enumerate(lemma_groups):
        lemma_frequency_dict = {}
        for lemma in group:
            lemma_frequency_dict[lemma] = lemma_frequency_dict.get(lemma, 0) + 1
        lemma_frequency_dicts.append(lemma_frequency_dict)

    # Створюємо DataFrame для частот лем
    lemma_df = pd.DataFrame(lemma_frequency_dicts).fillna(0).astype(int).T
    lemma_df.columns = [f'freq_{i + 1}' for i in range(num_samples)]

    # Додаємо колонку 'gen_freq' та рахуємо загальну частоту для кожного слова
    lemma_df.insert(0, 'gen_freq', lemma_df.sum(axis=1))

    # Замінюємо індекси на 'lemma'
    lemma_df.index.name = 'lemma'
    lemma_df.reset_index(inplace=True)
    lemma_df.index += 1

    # Записуємо DataFrame в базу даних
    lemma_df.to_sql('lemma_frequencies', conn, if_exists='replace', index=False)

    # Функція для отримання частини мови слова
    def get_pos(word):
        parsed_word = morph.parse(word)[0]
        return parsed_word.tag.POS

    # Застосовуємо функцію до всіх слів
    pos_tags = [get_pos(word) for word in non_latin_words]

    # Розбиваємо на 20 груп по 1000 тегів частин мови
    pos_groups = [pos_tags[i * sample_size: (i + 1) * sample_size] for i in range(num_samples)]

    # Створюємо частотний словник для кожної вибірки тегів частин мови
    pos_frequency_dicts = []
    for i, group in enumerate(pos_groups):
        pos_frequency_dict = {}
        for pos_tag in group:
            pos_frequency_dict[pos_tag] = pos_frequency_dict.get(pos_tag, 0) + 1
        pos_frequency_dicts.append(pos_frequency_dict)

    # Створюємо DataFrame для частот тегів частин мови
    pos_df = pd.DataFrame(pos_frequency_dicts).fillna(0).astype(int).T
    pos_df.columns = [f'freq_{i + 1}' for i in range(num_samples)]

    # Додаємо колонку 'gen_freq' та рахуємо загальну частоту для кожного слова
    pos_df.insert(0, 'gen_freq', pos_df.sum(axis=1))

    # Замінюємо індекси на 'pos_tag'
    pos_df.index.name = 'pos_tag'
    pos_df.reset_index(inplace=True)
    pos_df.index += 1

    # Застосовуємо функцію до всіх слів
    pos_tags = [get_pos(word) for word in non_latin_words]

    # Роздруковуємо слова за частинами мови
    print_words_by_pos(pos_tags, non_latin_words)

    # Записуємо DataFrame в базу даних
    pos_df.to_sql('pos_frequencies', conn, if_exists='replace', index=False)
    # Додаємо колонку 'total_pos_frequencies' до таблиці 'pos_frequencies'
    query = """
        ALTER TABLE pos_frequencies
        ADD COLUMN total_pos_frequencies INTEGER
    """

    conn.execute(query)

    # Підрахунок абсолютної частоти для кожної частини мови
    for i in range(1, num_samples + 1):
        query = f"""
            UPDATE pos_frequencies
            SET total_pos_frequencies = COALESCE(freq_{i}, 0) + COALESCE(total_pos_frequencies, 0)
        """
        conn.execute(query)

    # Розрахунок TF-IDF з обмеженням кількості фіч
    vectorizer = TfidfVectorizer(max_features=1000)  # Adjust the number as needed
    tfidf_matrix = vectorizer.fit_transform([' '.join(lemmas)])

    # Створюємо DataFrame для TF-IDF
    tfidf_df = pd.DataFrame(list(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])),
                            columns=['word', 'tfidf'])
    # Записуємо DataFrame в базу даних
    tfidf_df.to_sql('tfidf', conn, if_exists='replace', index=False)

    # Викликаємо метод select_word_avg_freq
    database_instance = DatabaseClass()
    database_instance.select_word_avg_freq()

    # Закриваємо з'єднання
    conn.close()
