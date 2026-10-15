import re
from collections import Counter
import pandas as pd
import math

stop_words = {
    'до', 'по', 'об', 'за', 'на', 'для', 'iii', '2025',
    'и', 'в', 'на', 'с', 'по', 'для', 'от', 'к', 'у', 'о', 'об', 'за', 'при', 'через',
    'со', 'то', 'как', 'что', 'это', 'или', 'а', 'но', 'да', 'же', 'бы', 'ли', 'ни', 'не',
    'был', 'была', 'было', 'были', 'есть', 'будет', 'может', 'ну', 'ах', 'ой', 'эх', 'ох',
    'уф', 'пфф', 'хм', 'мм', 'аа', 'ээ', 'ии'
}

# Письма
spam_emails = [
    "Только до конца дня – комплексный аудит информационной безопасности по суперцене.",
    "Внимание! Согласно новым требованиям, ваш сертификат по информационной безопасности утратил силу.",
    "Раскрываем рабочий метод обхода многофакторной аутентификации, который используют профессионалы."
]

ham_emails = [
    "Сообщаем об обновлении корпоративной политики информационной безопасности.",
    "Направляем результаты планового информационной безопасности аудит за III квартал 2025 года.",
    "Открыта регистрация на внутренний курс «Основы криптографии для специалистов информационной безопасности»."
]

test = "До конца дня открыта запись на аудит информационной безопасности со специальными условиями"

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Удаляем пунктуацию
    text = re.sub(r'\d+', '', text)       # Удаляем числа
    words = text.split()
    return [word for word in words if word and word not in stop_words]

spam_words = []
ham_words = []

for email in spam_emails:
    spam_words.extend(tokenize(email))

for email in ham_emails:
    ham_words.extend(tokenize(email))

# Подсчёт частот
spam_counter = Counter(spam_words)
ham_counter = Counter(ham_words)

# Все уникальные слова
all_words = set(spam_counter.keys()) | set(ham_counter.keys())

word_counts = {}
for word in sorted(all_words):
    word_counts[word] = (spam_counter[word], ham_counter[word])

# Исправление N_spam и N_ham: сумма частот, а не количество уникальных
N_spam = sum(c_s for _, (c_s, _) in word_counts.items())
N_ham = sum(c_h for _, (_, c_h) in word_counts.items())
V = len(word_counts)
a = 1

data = []
for word in word_counts:
    count_spam, count_ham = word_counts[word]
    p_spam = (count_spam + a) / (N_spam + V * a)
    p_ham = (count_ham + a) / (N_ham + V * a)
    data.append({
        'Слово': word,
        'Кол. вх. в спам': count_spam,
        'Кол. вх в не спам': count_ham,
        'P_с (A_j)': round(p_spam, 4),
        'P_н (A_j)': round(p_ham, 4)
    })

df = pd.DataFrame(data)
print("Таблица 2 – Расчет вероятностей появления слов")

P_H_spam = len(spam_emails) / (len(spam_emails) + len(ham_emails))
P_H_ham = len(ham_emails) / (len(spam_emails) + len(ham_emails))

test_words = tokenize(test)

new_words_set = set(test_words) - set(word_counts.keys())
new_words = sorted(list(new_words_set))
new_data = []
for word in new_words:
    count_spam = 0
    count_ham = 0
    p_spam = (count_spam + a) / (N_spam + V * a)
    p_ham = (count_ham + a) / (N_ham + V * a)
    new_data.append({
        'Слово': word,
        'Кол. вх. в спам': count_spam,
        'Кол. вх в не спам': count_ham,
        'P_с (A_j)': round(p_spam, 4),
        'P_н (A_j)': round(p_ham, 4)
    })

new_df = pd.DataFrame(new_data)
if not new_df.empty:
    print(new_df.to_string(index=False))

words_list = new_df['Слово'].tolist()

for w in words_list:
    word_counts[w] = (0, 0)

log_P_spam = math.log2(P_H_spam) + sum(math.log2((word_counts.get(word, (0, 0))[0] + a) / (N_spam + V * a)) for word in test_words)
log_P_ham = math.log2(P_H_ham) + sum(math.log2((word_counts.get(word, (0, 0))[1] + a) / (N_ham + V * a)) for word in test_words)

print(f"\nLog2 P(спам | test): {log_P_spam:.3f}")
print(f"Log2 P(не спам | test): {log_P_ham:.3f}")

if log_P_spam > log_P_ham:
    classification = "спам"
else:
    classification = "не спам"

print(f"\nВывод: сообщение '{test}' является {classification}.")
print(f"(Поскольку {log_P_spam:.3f} < {log_P_ham:.3f}, вероятность не-спама выше.)")

