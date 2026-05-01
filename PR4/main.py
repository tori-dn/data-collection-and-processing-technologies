import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


hours = np.arange(24)

prices = np.array([
    1.2, 1.2, 1.2, 1.2, 1.2, 1.5,   # ніч (дешевше)
    2.5, 3.0, 3.0, 3.0, 3.0, 3.0,   # ранок/день
    3.0, 3.0, 3.0, 3.5, 4.0, 5.0,   # передпік
    8.0, 9.0, 8.5,                  # вечірній пік
    4.0, 2.0, 1.5                   # вечірній спад
])

total_energy_needed = 180.0  # необхідна енергія за добу (кВт*год)
p_min = 2.0                  # мінімальне споживання
p_max = 15.0                 # максимальна потужність мережі


# Виконання зроблено так, що людина знає, що вночі дешевше, тому ставитиме більше навантаження там, але не враховує точні піки цін ввечері.

p_heuristic = np.full(24, 5.3) # пазове денне споживання
p_heuristic[0:7] = 12.8        # підвищене нічне споживання
p_heuristic = p_heuristic * (total_energy_needed / np.sum(p_heuristic))


def objective(p):
    return np.sum(p * prices)

# сумарна енергія має дорівнювати 180 кВт*год
def constraint_energy(p):
    return np.sum(p) - total_energy_needed

cons = ({'type': 'eq', 'fun': constraint_energy})

bounds = [(p_min, p_max) for _ in range(24)]

# початкове наближення (рівномірний розподіл)
p0 = np.full(24, total_energy_needed / 24)

# Запуск оптимізатора SLSQP
res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=cons)
p_opt = res.x


cost_heuristic = np.sum(p_heuristic * prices)
cost_optimized = res.fun
savings = cost_heuristic - cost_optimized

print("\n" + "="*65)
print(f"{'Година':<7} | {'Ціна':<8} | {'Евристика (кВт)':<16} | {'SciPy (кВт)':<12}")
print("-" * 65)
for h in range(24):
    print(f"{h:02d}:00   | {prices[h]:<8.2f} | {p_heuristic[h]:<16.2f} | {p_opt[h]:<12.2f}")

print("-" * 65)
print(f"Загальні витрати (Евристика): {cost_heuristic:.2f} грн")
print(f"Загальні витрати (SciPy):     {cost_optimized:.2f} грн")
print(f"ЧИСТА ЕКОНОМІЯ:               {savings:.2f} грн/добу")
print("="*65 + "\n")


fig, ax1 = plt.subplots(figsize=(14, 7))

ax2 = ax1.twinx()
ax2.bar(hours, prices, alpha=0.2, color='orange', width=0.9, label='Ціна електроенергії (грн/кВт*год)')
ax2.set_ylabel('Ціна (грн)', color='orange', fontsize=12)
ax2.set_ylim(0, 10)

ax1.plot(hours, p_heuristic, color='#e63946', marker='o', linestyle='--', label='Евристика "На око"', linewidth=2)
ax1.plot(hours, p_opt, color='#2a9d8f', marker='s', linestyle='-', label='SciPy Оптимізація', linewidth=3)

ax1.text(1, 14, f'Збережено: {savings:.2f} грн/добу', fontsize=12, fontweight='bold',
         bbox=dict(facecolor='green', alpha=0.4, boxstyle='round,pad=0.5', ec='black'))

ax1.set_title('Оптимізація енерговитрат: Математичний підхід vs Інтуїтивний', fontsize=16, pad=20)
ax1.set_xlabel('Година доби', fontsize=12)
ax1.set_ylabel('Споживання потужності (кВт)', fontsize=12)
ax1.set_xticks(hours)
ax1.set_ylim(0, 16)
ax1.grid(True, linestyle=':', alpha=0.6)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.show()