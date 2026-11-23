from llm_helper import ask_llm, save_report


def run():
    output = ["Завдання 4: Використання системної ролі\n"]

    task_prompt = "Поясни принцип наслідування в ООП."

    roles = [
        (
            "Викладач",
            "Ти — університетський викладач. Пояснюй матеріал просто, доброзичливо, використовуючи аналогії з реального життя.",
        ),
        (
            "Програміст",
            "Ти — Senior Python Developer. Відповідай лаконічно, технічною мовою, роблячи акцент на перевикористанні коду та наводячи короткий приклад класів.",
        ),
        (
            "Науковець",
            "Ти — науковець у галузі Computer Science. Використовуй сувору академічну термінологію, згадуючи теорію типів та ієрархію абстракцій.",
        ),
    ]

    for role_name, sys_instruction in roles:
        response = ask_llm(task_prompt, system_instruction=sys_instruction)
        block = (
            f"--- Роль: {role_name} ---\n"
            f"System Instruction: {sys_instruction}\n"
            f"Prompt: {task_prompt}\n\n"
            f"Відповідь:\n{response}\n"
        )
        print(block)
        output.append(block)

    save_report("result_task4.txt", output)


if __name__ == "__main__":
    run()