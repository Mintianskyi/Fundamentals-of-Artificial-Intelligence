import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.title("Мережі Хопфілда і Хемінга")

st.sidebar.header("Налаштування")
net_choice = st.sidebar.selectbox("Оберіть мережу:", ("Мережа Хопфілда", "Мережа Хемінга"))
cols = st.sidebar.number_input("Стовпчики (n)", min_value=1, value=4)
rows = st.sidebar.number_input("Рядки (m)", min_value=1, value=5)
threshold = st.sidebar.slider("Поріг", 0, 255, 128)

def get_norm(file, nc, nr, th):
    img = Image.open(file).convert("L")
    w, h = img.size
    arr = np.array(img) < th
    cw, ch = w / nc, h / nr
    vec = []
    for r in range(nr):
        for c in range(nc):
            vec.append(np.sum(arr[int(r*ch):int((r+1)*ch), int(c*cw):int((c+1)*cw)]))
    vec = np.array(vec, dtype=float)
    tot = np.sum(vec)
    return vec / tot if tot > 0 else np.zeros_like(vec)

def get_bipolar(file, nc, nr, th):
    img = Image.open(file).convert("L")
    w, h = img.size
    arr = np.array(img) < th
    cw, ch = w / nc, h / nr
    vec = []
    for r in range(nr):
        for c in range(nc):
            cnt = np.sum(arr[int(r*ch):int((r+1)*ch), int(c*cw):int((c+1)*cw)])
            vec.append(1 if cnt > (cw * ch / 4) else -1)
    return np.array(vec, dtype=float)

st.subheader("Завантаження еталонів (3 класи)")
tabs = st.tabs(["Клас 1", "Клас 2", "Клас 3"])
patterns = []

for i, t in enumerate(tabs):
    with t:
        file = st.file_uploader(f"Еталон {i+1}", type=["png", "jpg", "bmp"], key=f"f_{i}")
        if file is not None:
            st.image(file, width=100)
            if "Хопфілда" in net_choice:
                patterns.append(get_bipolar(file, cols, rows, threshold))
            else:
                patterns.append(get_norm(file, cols, rows, threshold))

st.divider()

if len(patterns) == 3:
    if "Хопфілда" in net_choice:
        st.subheader("Мережа Хопфілда")
        n_neu = cols * rows
        
        W = np.zeros((n_neu, n_neu))
        for p in patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0)
        
        with st.expander("Подивитись матрицю вагів W"):
            st.dataframe(pd.DataFrame(W))
            
        test_f = st.file_uploader("Тестовий (зашумлений) файл", type=["png", "jpg", "bmp"], key="th_test")
        if test_f is not None:
            y = get_bipolar(test_f, cols, rows, threshold)
            for _ in range(15):
                net = np.dot(W, y)
                new_y = np.where(net >= 0, 1, -1)
                if np.array_equal(y, new_y):
                    break
                y = new_y
            sims = [np.dot(y, p) for p in patterns]
            res_cls = np.argmax(sims) + 1
            st.success(f"Мережа Хопфілда впізнала образ: **Клас {res_cls}**")
            
    else:
        st.subheader("Мережа Хемінга")
        test_f = st.file_uploader("Тестовий файл для Хемінга", type=["png", "jpg", "bmp"], key="tm_test")
        if test_f is not None:
            x_test = get_norm(test_f, cols, rows, threshold)
            X_mat = np.array(patterns)
            m = 3
            n = cols * rows
            
            W1 = X_mat.T / 2.0
            B1 = np.ones(m) * (n / 2.0)
            Y1 = np.dot(x_test, W1) + B1
            
            Y2 = Y1.copy()
            eps = 0.1 / m
            for _ in range(25):
                sum_others = np.sum(Y2) - Y2
                Y2_new = np.maximum(0, Y2 - eps * sum_others)
                if np.array_equal(Y2, Y2_new):
                    break
                Y2 = Y2_new
                
            win = np.argmax(Y2) + 1
            st.success(f"Мережа Хемінга визначила: **Клас {win}** (активація: {Y2[win-1]:.4f})")
else:
    st.info("Будь ласка, завантажте всі 3 еталони вище.")