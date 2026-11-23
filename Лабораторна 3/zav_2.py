from llm_helper import ask_llm, save_report


def run():
    output = ["Завдання 2. Порівняння Zero-shot та Few-shot\n"]

    zero_shot_prompt = (
        "Напиши SQL-запит для таблиці Tovar (id, name, price), "
        "щоб вибрати товари дорожче 500 грн. Виведи тільки SQL-код."
    )

    few_shot_prompt = (
        "Перетвори запит у SQL для таблиці Tovar (id, name, price). Виводь тільки SQL-код.\n"
        "Приклад 1: Вибрати товари дорожче 100 -> SELECT * FROM Tovar WHERE price > 100;\n"
        "Приклад 2: Вибрати товари дешевше 50 -> SELECT * FROM Tovar WHERE price < 50;\n"
        "Завдання: Вибрати товари дорожче 500 грн ->"
    )

    for approach, prompt in [("Zero-shot", zero_shot_prompt), ("Few-shot", few_shot_prompt)]:
        block = f"--- {approach} ---\nPrompt:\n{prompt}\n\nВідповідь:\n{ask_llm(prompt)}\n"
        print(block)
        output.append(block)

    save_report("result_task2.txt", output)


if __name__ == "__main__":
    run()