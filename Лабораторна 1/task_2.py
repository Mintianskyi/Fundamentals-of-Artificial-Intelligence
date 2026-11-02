import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="task_2", layout="wide")

st.title("Розпізнавання за методом еталонів")

st.sidebar.header("Параметри")
num_classes = st.sidebar.selectbox("Кількість класів", [3, 4])
cols = st.sidebar.number_input("Стовпчики (n)", min_value=1, value=4)
rows = st.sidebar.number_input("Рядки (m)", min_value=1, value=5)
threshold = st.sidebar.slider("Поріг темного", 0, 255, 128)

metric = st.sidebar.selectbox(
    "Метрика відстані",
    ["Евклідова норма", "Норма Чебишева", "Манхеттенська норма"]
)

def get_features(file, n_cols, n_rows, thresh):
    img = Image.open(file).convert("RGB")
    gray = img.convert("L")
    w, h = img.size
    
    arr = np.array(gray)
    mask = arr < thresh
    
    cell_w = w / n_cols
    cell_h = h / n_rows
    
    abs_vec = []
    for r in range(n_rows):
        for c in range(n_cols):
            y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
            x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
            count = np.sum(mask[y1:y2, x1:x2])
            abs_vec.append(count)
            
    abs_vec = np.array(abs_vec, dtype=float)
    total = np.sum(abs_vec)
    norm_vec = abs_vec / total if total > 0 else np.zeros_like(abs_vec)
    
    grid_img = img.copy()
    draw = ImageDraw.Draw(grid_img)
    for c in range(1, n_cols):
        draw.line([(int(c * cell_w), 0), (int(c * cell_w), h)], fill="red", width=2)
    for r in range(1, n_rows):
        draw.line([(0, int(r * cell_h)), (w, int(r * cell_h))], fill="red", width=2)
        
    return abs_vec, norm_vec, grid_img, total


def calc_dist(v1, v2, m_type):
    if m_type == "Евклідова норма":
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    elif m_type == "Норма Чебишева":
        return float(np.max(np.abs(v1 - v2)))
    elif m_type == "Манхеттенська норма":
        return float(np.sum(np.abs(v1 - v2)))
    return 0.0


st.subheader("1. Еталони класів")
tabs = st.tabs([f"Клас {i + 1}" for i in range(num_classes)])

templates = []
ready = True

for i in range(num_classes):
    with tabs[i]:
        name = st.text_input(f"Назва класу {i + 1}", value=f"Клас {i + 1}", key=f"name_{i}")
        file = st.file_uploader(f"Файл для {name}", type=["bmp", "png", "jpg", "jpeg"], key=f"f_{i}")
        
        if file is not None:
            abs_v, norm_v, g_img, tot = get_features(file, cols, rows, threshold)
            templates.append({
                "name": name,
                "file_name": file.name,
                "norm_vec": norm_v,
                "abs_vec": abs_v
            })
            
            st.image(g_img, caption=f"Еталон {name}", width=250)
            st.write(f"**Абсолютний вектор:** {list(abs_v.astype(int))}")
            st.write(f"**Нормований вектор:** {[round(x, 4) for x in norm_v]}")
        else:
            ready = False
            st.warning("Завантажте файл еталону")

st.divider()

st.subheader("2. Невідомий образ")
unknown_file = st.file_uploader("Завантажте тестове зображення", type=["bmp", "png", "jpg", "jpeg"])

if unknown_file is not None and ready:
    unk_abs, unk_norm, unk_grid, unk_tot = get_features(unknown_file, cols, rows, threshold)
    
    st.image(unk_grid, caption="Невідомий образ", width=250)
    st.write(f"**Абсолютний вектор:** {list(unk_abs.astype(int))}")
    st.write(f"**Нормований вектор:** {[round(x, 4) for x in unk_norm]}")
    
    st.divider()
    
    st.subheader(f"3. Порівняння ({metric})")
    results = []
    
    for t in templates:
        d = calc_dist(unk_norm, t["norm_vec"], metric)
        results.append({
            "Клас": t["name"],
            "Файл еталону": t["file_name"],
            "Відстань": round(d, 5)
        })
        
    df = pd.DataFrame(results).sort_values(by="Відстань")
    st.dataframe(df, use_container_width=True)
    
    best = df.iloc[0]
    st.success(f"**Рішення:** Образ віднесено до **{best['Клас']}** (відстань = {best['Відстань']})")

    text_report = f"Результат класифікації ({metric}): {best['Клас']}\n\nВідстані:\n"
    for _, row in df.iterrows():
        text_report += f"{row['Клас']}: {row['Відстань']}\n"

    st.download_button("Завантажити звіт (.txt)", text_report, file_name="result_lab2.txt")

elif not ready:
    st.info("Спочатку завантажте всі еталони вище.")