import numpy as np
import time
from numba import njit
from tabulate import tabulate


@njit
def numba_sort(arr):
    # .copy() потрібен, щоб не змінювати вхідний масив (аналогічно до np.sort)
    res = arr.copy()
    res.sort()
    return res


def run_assignment():
    SIZE = 15_000_000
    print(f"\n Аналіз енергетичних даних ({SIZE:,} записів)")
    print("=" * 60)

    data = np.random.uniform(0.5, 5.0, size=SIZE).astype(np.float64)

    start = time.perf_counter()
    np_res = np.sort(data)
    t_numpy = time.perf_counter() - start

    start = time.perf_counter()
    numba_sort(data)
    t_numba_cold = time.perf_counter() - start

    start = time.perf_counter()
    nb_res = numba_sort(data)
    t_numba_hot = time.perf_counter() - start

    table = [
        ["Библіотека / Метод", "Час (сек)", "Продуктивність"],
        ["NumPy (np.sort)", f"{t_numpy:.4f}", "100% (Base)"],
        ["Numba (Cold Start)", f"{t_numba_cold:.4f}", f"{(t_numpy / t_numba_cold) * 100:.1f}%"],
        ["Numba (Hot Start)", f"{t_numba_hot:.4f}", f"{(t_numpy / t_numba_hot) * 100:.1f}%"]
    ]

    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

    print("\n Оцінка результатів:")
    print(f"1. Сортування {SIZE} елементів NumPy виконує стабільно за рахунок C-оптимізації.")
    print(f"2. Numba при першому запуску повільніша через JIT-компіляцію.")

    if t_numba_hot < t_numpy:
        diff = (t_numpy / t_numba_hot)
        print(f"3. ВИСНОВОК: Numba виявилася швидшою у {diff:.2f} разів. Це ефективно для Pipeline обробки.")
    else:
        print(
            f"3. ВИСНОВОК: NumPy залишається швидшим для чистого сортування. Numba доцільна лише у комплексних алгоритмах.")


if __name__ == "__main__":
    run_assignment()
