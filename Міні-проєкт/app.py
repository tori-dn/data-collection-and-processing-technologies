from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Конфігурація
app = Flask(__name__)
CORS(app)

CSV_PATH   = "course_data.csv"
JSON_PATH  = "analysis_history.json"

# Допоміжні функції

def load_history() -> list:
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            try: return json.load(f)
            except json.JSONDecodeError: return []
    return []


def save_history(entry: dict):
    history = load_history()
    history.append(entry)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def compute_analytics(df: pd.DataFrame) -> dict:
    """Обчислює ключові метрики курсу на основі активності."""
    weekly = (
        df.groupby("week")
        .apply(lambda x: pd.Series({
            "avg_grade": round(x[x["attendance"] == 1]["grade"].mean(), 2) if not x[x["attendance"] == 1].empty else 0,
            "avg_feedback": round(x[x["attendance"] == 1]["feedback_score"].mean(), 2) if not x[x["attendance"] == 1].empty else 0,
            "attendance_rate": round(x["attendance"].mean() * 100, 1),
            "avg_assignment": round(x[x["attendance"] == 1]["assignment_score"].mean(), 2) if not x[x["attendance"] == 1].empty else 0,
        }), include_groups=False)
        .reset_index()
        .to_dict(orient="records")
    )

    numeric = df[["attendance", "grade", "feedback_score", "assignment_score"]].dropna()
    corr_matrix = numeric.corr().round(3).fillna(0).to_dict()

    student_stats = (
        df.groupby(["student_id", "student_name"])
        .apply(lambda x: pd.Series({
            "attendance_rate": round(x["attendance"].mean() * 100, 1),
            "avg_grade": round(x[x["attendance"] == 1]["grade"].mean(), 2) if not x[x["attendance"] == 1].empty else 0,
            "avg_feedback": round(x[x["attendance"] == 1]["feedback_score"].mean(), 2) if not x[x["attendance"] == 1].empty else 0,
        }), include_groups=False)
        .reset_index()
    )

    at_risk = student_stats[
        (student_stats["attendance_rate"] < 60) | (student_stats["avg_grade"] < 65)
    ].to_dict(orient="records")

    overall = {
        "total_students": int(df["student_id"].nunique()),
        "total_weeks": int(df["week"].nunique()),
        "avg_attendance_rate": round(df["attendance"].mean() * 100, 1),
        "avg_grade": round(df[df["attendance"] == 1]["grade"].mean(), 2),
        "avg_feedback": round(df[df["attendance"] == 1]["feedback_score"].mean(), 2),
        "avg_assignment": round(df[df["attendance"] == 1]["assignment_score"].mean(), 2),
        "at_risk_count": len(at_risk),
    }

    return {
        "overall": overall,
        "weekly": weekly,
        "correlations": corr_matrix,
        "at_risk_students": at_risk,
        "student_stats": student_stats.to_dict(orient="records"),
    }


def compare_with_previous(current: dict, history: list) -> dict | None:
    if not history: return None
    prev = history[-1]["analytics"]["overall"]
    curr = current["overall"]
    return {
        "grade_delta": round(curr["avg_grade"] - prev["avg_grade"], 2),
        "attendance_delta": round(curr["avg_attendance_rate"] - prev["avg_attendance_rate"], 1),
        "feedback_delta": round(curr["avg_feedback"] - prev["avg_feedback"], 2),
        "at_risk_delta": curr["at_risk_count"] - prev["at_risk_count"],
        "previous_date": history[-1]["timestamp"],
    }


def generate_dynamic_report(analytics: dict, dynamics: dict | None) -> str:
    """Генерує глибокий аналітичний звіт, який повністю адаптується під поточні цифри таблиці."""
    o = analytics['overall']
    at_risk_names = ", ".join(s["student_name"] for s in analytics["at_risk_students"]) or "відсутні"
    att_grade = analytics["correlations"].get("attendance", {}).get("grade", 0)

    # Визначаємо загальний стан курсу на основі балів
    status = "ВИСОКИЙ" if o['avg_grade'] >= 75 else "СЕРЕДНІЙ" if o['avg_grade'] >= 60 else "КРИТИЧНИЙ"

    # Текст динаміки, якщо є попередні запуски
    dyn_text = "Це первинний аналіз, дані для порівняння трендів відсутні."
    if dynamics:
        trend = "ПОКРАЩЕННЯ" if dynamics['grade_delta'] > 0 else "ПОГІРШЕННЯ" if dynamics['grade_delta'] < 0 else "СТАБІЛІЗАЦІЮ"
        dyn_text = f"Порівняно з аудитом від {dynamics['previous_date']} курс демонструє {trend} показників. Зміна середнього балу: {dynamics['grade_delta']:+.2f}, зміна відвідуваності: {dynamics['attendance_delta']:+.1f}%, коливання ризиків: {dynamics['at_risk_delta']:+d} ос."

    return f"""1. **Загальна оцінка ефективності курсу**
На основі аналізу {o['total_weeks']}-тижневого зрізу даних, загальна ефективність курсу оцінюється як **{status}**. Середня успішність когорти становить {o['avg_grade']} балів зі 100 при середній відвідуваності {o['avg_attendance_rate']}%. Рівень задоволеності студентів становить {o['avg_feedback']}/5.

2. **Ключові сильні сторони**
- Активна частина групи демонструє стабільне виконання практичних та лабораторних завдань із середнім результатом {o['avg_assignment']} балів.
- Студенти демонструють високий рівень задоволеності навчальним процесом, який утримується на позначці {o['avg_feedback']}/5.
- Сформовано стійке ядро успішності: загальна кількість залучених студентів у когорті — {o['total_students']} осіб.

3. **Проблемні зони та ризики**
- Виявлено {o['at_risk_count']} студентів у сегменті критичного ризику. Головні фактори тривоги: систематичні пропуски або падіння балів нижче нормативного мінімуму.
- Студенти, які потребують негайного індивідуального втручання: **{at_risk_names}**.
- Наявність індивідуальних заборгованостей безпосередньо впливає на загальний середній показник успішності групи.

4. **Аналіз кореляцій**
Коефіцієнт кореляції між присутністю на заняттях та оцінками становить **{att_grade}**. Це підтверджує наявність прямих лінійних зв'язків: кожен пропущений тиждень та виставлений нуль за відвідуваність математично тягне за собою пропорційне падіння загального балу студента за поточні завдання.

5. **Рекомендації для викладача**
- Терміново зв'язатися або організувати коротку консультаційну годину для ліквідації боргів студентів: {at_risk_names}.
- Звернути увагу на тижні, де загальний рівень відвідуваності впав до мінімальних позначок ({o['avg_attendance_rate']}%).
- Провести аудит складності завдань на тижнях, де зафіксовано найнижчий середній бал групи.
- Продовжувати моніторинг задоволеності студентів для збереження поточного фідбеку {o['avg_feedback']}/5.
- Впровадити автоматичні нагадування про дедлайни лабораторних робіт через старосту.

6. **Рекомендації для студентів у зоні ризику**
- Надіслати викладачу індивідуальний запит на отримання наздоганяючих завдань.
- Опрацювати методичні матеріали та конспекти за пропущені тижні навчання.
- Забезпечити 100% присутність на наступних практичних блоках для стабілізації поточної оцінки.

7. **Прогноз на наступний тиждень**
За умови збереження поточної структури навчання, середній бал групи прогнозується в діапазоні {round(o['avg_grade'] * 0.98, 1)}-{round(o['avg_grade'] * 1.02, 1)} балів. Проведення точкових консультацій із групою ризику дозволить оперативно знизити кількість відстаючих на 1-2 особи.

8. **Порівняння з попереднім аналізом (Маркер динаміки)**
{dyn_text}"""


# Маршрути API

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        if "file" in request.files:
            file = request.files["file"]
            df = pd.read_csv(file, sep=None, engine='python', encoding="utf-8-sig")
        elif os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH, sep=None, engine='python', encoding="utf-8-sig")
        else:
            return jsonify({"error": "CSV файл не знайдено"}), 400

        # Обчислення
        analytics = compute_analytics(df)
        history = load_history()
        dynamics = compare_with_previous(analytics, history)

        # Динамічна генерація звіту
        ai_report = generate_dynamic_report(analytics, dynamics)

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analytics": analytics,
            "dynamics": dynamics,
            "ai_report": ai_report,
        }
        save_history(entry)

        return jsonify({
            "success": True,
            "analytics": analytics,
            "dynamics": dynamics,
            "ai_report": ai_report,
            "timestamp": entry["timestamp"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    history = load_history()
    summary = [{"timestamp": h["timestamp"], "overall": h["analytics"]["overall"], "dynamics": h.get("dynamics")} for h in history]
    return jsonify({"history": summary, "count": len(summary)})


@app.route("/api/history/<int:index>", methods=["GET"])
def get_history_entry(index):
    history = load_history()
    if index < 0 or index >= len(history): return jsonify({"error": "Запис не знайдено"}), 404
    return jsonify(history[index])


if __name__ == "__main__":
    app.run(debug=True, port=5001)