from llm_helper import ask_llm, save_report


def run():
    output = ["Завдання 3. Удосконалення prompt\n"]

    basic_prompt = "Розкажи про бази даних."

    improved_prompt = (
        "Ти — викладач баз даних. Поясни студенту 2 курсу спеціальності 'Інженерія програмного забезпечення', "
        "що таке реляційні бази даних. Обсяг: до 5 речень. "
        "Формат: структурований текст із виділенням ключових термінів."
    )

    for label, prompt in [("Початковий prompt", basic_prompt), ("Вдосконалений prompt", improved_prompt)]:
        block = f"--- {label} ---\nPrompt: {prompt}\n\nВідповідь:\n{ask_llm(prompt)}\n"
        print(block)
        output.append(block)

    save_report("result_task3.txt", output)


if __name__ == "__main__":
    run()