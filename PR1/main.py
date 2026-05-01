import pandas as pd
import numpy as np

try:
    df = pd.read_csv('products.csv')
except FileNotFoundError:
    print("Помилка: Файл 'products.csv' не знайдено!")
    exit()

df['correct_total'] = df['quantity'] * df['price'] * (1 - df['discount'])
df['abs_diff'] = abs(df['total'] - df['correct_total'])
mean_abs_diff = df['abs_diff'].mean()

mask = ~np.isclose(df['total'], df['correct_total'])
mismatches = df[mask].copy()


def print_separator(char="-", length=70):
    print(char * length)

print("\nЗВІТ ПРО ПЕРЕВІРКУ:")
print_separator("=")

if not mismatches.empty:
    header = f"{'Продукт':<15} | {'Надано':<15} | {'Має бути':<15} | {'Помилка':<10}"
    print(header)
    print_separator("-")
    for _, row in mismatches.iterrows():
        print(f"{row['product']:<15} | {row['total']:<15.2f} | {row['correct_total']:<15.2f} | {row['abs_diff']:<10.2f}")
else:
    print("Всі розрахунки у файлі коректні.")

print_separator("-")
print(f"{'СЕРЕДНЯ АБСОЛЮТНА РІЗНИЦЯ:':<48} {mean_abs_diff:>10.2f}")
print_separator("=")


df['total'] = df['correct_total']

df.drop(columns=['correct_total', 'abs_diff']).to_csv('fixed_products.csv', index=False)

print("Результат: Виправлені дані збережено у файл 'fixed_products.csv'\n")