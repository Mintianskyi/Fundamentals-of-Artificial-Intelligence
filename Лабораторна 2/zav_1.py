import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.title("Перцептрон Розенблата")

st.sidebar.header("Параметри")
cols = st.sidebar.number_input("Кількість стовпчиків (n)", min_value=1, value=4)
rows = st.sidebar.number_input("Кількість рядків (m)", min_value=1, value=5)
threshold = st.sidebar.slider("Поріг чорно-білого", 0, 255, 128)
lr = st.sidebar.number_input("Швидкість навчання (lr)", min_value=0.01, max_value=1.0, value=0.1)
epochs = st.sidebar.slider("Епохи", 1, 30, 10)

def get_features(file, n_cols, n_rows, thresh):
    img = Image.open(file).convert("L")
    w, h = img.size
    arr = np.array(img) < thresh
    cw, ch = w / n_cols, h / n_rows
    
    vec = []
    for r in range(n_rows):
        for c in range(n_cols):
            y1, y2 = int(r * ch), int((r + 1) * ch)
            x1, x2 = int(c * cw), int((c + 1) * cw)
            vec.append(np.sum(arr[y1:y2, x1:x2]))
            
    vec = np.array(vec, dtype=float)
    total = np.sum(vec)
    return vec / total if total > 0 else np.zeros_like(vec)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Клас А (мітка: +1)")
    files_a = st.file_uploader("Зразки для А", type=["png", "jpg", "bmp"], accept_multiple_files=True, key="a")
with c2:
    st.subheader("Клас Б (мітка: -1)")
    files_b = st.file_uploader("Зразки для Б", type=["png", "jpg", "bmp"], accept_multiple_files=True, key="b")

if files_a and files_b:
    dataset = []
    for f in files_a:
        dataset.append((get_features(f, cols, rows, threshold), 1))
    for f in files_b:
        dataset.append((get_features(f, cols, rows, threshold), -1))
        
    n_feat = cols * rows
    
    if "weights" not in st.session_state:
        st.session_state.weights = np.random.uniform(0.0, 0.2, n_feat)
        st.session_state.bias = 0.1

    if st.button("Запустити навчання перцептрона"):
        w = st.session_state.weights
        b = st.session_state.bias
        
        history_err = []
        for ep in range(epochs):
            err_count = 0
            for x, target in dataset:
                net = np.dot(w, x) - b
                out = 1 if net > 0 else -1
                
                if out != target:
                    w += lr * (target - out) * x
                    b -= lr * (target - out)
                    err_count += 1
            history_err.append(err_count)
            if err_count == 0:
                break
                
        st.session_state.weights = w
        st.session_state.bias = b
        st.success(f"Кроків (епох): {len(history_err)}. Помилок у кінці: {history_err[-1]}")
        st.write("Отримані ваги синапсів:", np.round(w, 4))

    st.subheader("Перевірка")
    test_img = st.file_uploader("Завантажте тестову картинку", type=["png", "jpg", "bmp"], key="t")
    if test_img is not None:
        x_t = get_features(test_img, cols, rows, threshold)
        net = np.dot(st.session_state.weights, x_t) - st.session_state.bias
        res = 1 if net > 0 else -1
        name = "Клас А (+1)" if res == 1 else "Клас Б (-1)"
        st.info(f"Результат розпізнавання: **{name}**")
else:
    st.warning("Завантажте файли для обох класів вище.")