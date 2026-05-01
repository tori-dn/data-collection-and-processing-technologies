import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


def setup_nltk():
    print("Перевірка та завантаження ресурсів NLTK...")
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    for res in resources:
        nltk.download(res, quiet=True)
    print("Ресурси готові до роботи.\n")


def process_text_file(input_filename):

    if not os.path.exists(input_filename):
        print(f"Помилка: Файл '{input_filename}' не знайдено!")
        return

    with open(input_filename, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    print(f"--- Оригінальний текст (перші 100 символів) ---")
    print(raw_text[:100] + "...")

    text = raw_text.lower()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))


    punctuation = set(string.punctuation)

    filtered_words = [
        word for word in tokens
        if word not in punctuation
           and word not in stop_words
           and word.isalpha()
    ]

    lemmatizer = WordNetLemmatizer()
    lemmatized_output = [lemmatizer.lemmatize(w) for w in filtered_words]

    frequency_dict = Counter(lemmatized_output)

    return frequency_dict, lemmatized_output


def main():
    setup_nltk()

    filename = "input.txt"
    result = process_text_file(filename)

    if result:
        freq_dict, cleaned_list = result

        print(f"\n--- Результати аналізу ---")
        print(f"Кількість токенів після очищення: {len(cleaned_list)}")
        print(f"Кількість унікальних лем: {len(freq_dict)}")

        print("\nТоп-10 найчастіших слів у тексті:")
        print(f"{'Слово':<15} | {'Частота':<10}")
        print("-" * 28)

        for word, count in freq_dict.most_common(10):
            print(f"{word:<15} | {count:<10}")

        with open("output_frequency.txt", "w", encoding="utf-8") as out_f:
            out_f.write("Word,Frequency\n")
            for word, count in freq_dict.most_common():
                out_f.write(f"{word},{count}\n")
        print("\nПовний частотний словник збережено у файл 'output_frequency.txt'")


if __name__ == "__main__":
    main()
