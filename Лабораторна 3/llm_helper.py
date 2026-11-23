import os
from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

MODEL_ID = "gemini-3.8-flash"

def ask_llm(prompt: str, system_instruction: str = None) -> str:
    config = (
        types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
        )
        if system_instruction
        else None
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config,
    )
    return response.text.strip()

def save_report(filename: str, lines: list[str]):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Звіт успішно збережено у файл: {file_path}")