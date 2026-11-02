import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="task_1", layout="wide")

st.title("Побудова векторів ознак")

st.sidebar.header("Параметри")
cols = st.sidebar.number_input("Стовпчики (n)", min_value=1, value=4)
rows = st.sidebar.number_input("Рядки (m)", min_value=1, value=5)
threshold = st.sidebar.slider("Поріг темного", 0, 255, 128)

file = st.file_uploader("Завантажте картинку", type=["bmp", "png", "jpg", "jpeg"])

if file is not None:
    orig_img = Image.open(file).convert("RGB")
    gray_img = orig_img.convert("L")
    w, h = orig_img.size
    
    img_arr = np.array(gray_img)
    mask = img_arr < threshold

    cell_w = w / cols
    cell_h = h / rows
    
    abs_vector = []
    
    for r in range(rows):
        for c in range(cols):
            y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
            x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
            
            count = np.sum(mask[y1:y2, x1:x2])
            abs_vector.append(count)
            
    abs_vector = np.array(abs_vector)
    
    total = np.sum(abs_vector)
    norm_vector = abs_vector / total if total > 0 else np.zeros_like(abs_vector, dtype=float)

    grid_img = orig_img.copy()
    draw = ImageDraw.Draw(grid_img)
    
    for c in range(1, cols):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, h)], fill="red", width=2)

    for r in range(1, rows):
        y = int(r * cell_h)
        draw.line([(0, y), (w, y)], fill="red", width=2)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Картинка з сіткою")
        st.image(grid_img, caption=f"Розмір: {w}x{h} px | Сітка: {cols}x{rows}", use_container_width=True)

    with col2:
        st.subheader("Результати")
        
        st.write(f"**Всього темних пікселів:** {total}")
        
        st.write("**Абсолютний вектор (X):**")
        st.write(list(abs_vector))
        
        st.write("**Нормований вектор (X_norm):**")
        st.write([round(x, 4) for x in norm_vector])
        
        df = pd.DataFrame({
            "Комірка": [f"({r+1},{c+1})" for r in range(rows) for c in range(cols)],
            "X": abs_vector,
            "X_norm": np.round(norm_vector, 4)
        })
        st.dataframe(df, use_container_width=True)

        res_text = f"X = {list(abs_vector)}\nX_norm = {[round(x, 4) for x in norm_vector]}"
        st.download_button("Зберегти у txt", res_text, file_name="result.txt")
else:
    st.info("Завантажте файл зображення.")