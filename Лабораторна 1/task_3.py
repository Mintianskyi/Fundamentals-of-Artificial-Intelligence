import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="task_3", layout="wide")

st.title("Розпізнавання образів з навчанням")

st.sidebar.header("Параметри")
cols = st.sidebar.number_input("Стовпчики (n)", min_value=1, value=4)
rows = st.sidebar.number_input("Рядки (m)", min_value=1, value=5)
threshold = st.sidebar.slider("Поріг темного", 0, 255, 128)

method = st.sidebar.radio(
    "Метод аналізу:",
    ("1) Геометричні центри (матсподівання)", "2) Ортогональні області [Min, Max]")
)

def get_features(file, n_cols, n_rows, thresh):
    img = Image.open(file).convert("RGB")
    gray = img.convert("L")
    w, h = img.size

    arr = np.array(gray)
    mask = arr < thresh

    cell_w, cell_h = w / n_cols, h / n_rows

    abs_vec = []
    for r in range(n_rows):
        for c in range(n_cols):
            y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
            x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
            abs_vec.append(np.sum(mask[y1:y2, x1:x2]))

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


st.subheader("1. Навчання (3 класи)")
tabs = st.tabs(["Клас 1", "Клас 2", "Клас 3"])

classes_data = {}
ready = True

for i, tab in enumerate(tabs):
    with tab:
        name = st.text_input(f"Назва класу {i + 1}", value=f"Клас {i + 1}", key=f"cname_{i}")
        files = st.file_uploader(
            f"Зразки для {name}",
            type=["bmp", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"cfiles_{i}"
        )

        if files:
            norm_list = []
            preview_cols = st.columns(min(len(files), 4))
            
            for idx, f in enumerate(files):
                _, norm_v, g_img, _ = get_features(f, cols, rows, threshold)
                norm_list.append(norm_v)
                if idx < 4:
                    with preview_cols[idx]:
                        st.image(g_img, caption=f.name, width=120)

            norm_matrix = np.array(norm_list)

            classes_data[name] = {
                "mean": np.mean(norm_matrix, axis=0),
                "min": np.min(norm_matrix, axis=0),
                "max": np.max(norm_matrix, axis=0),
                "count": len(files)
            }

            st.write(f"Завантажено зразків: **{len(files)}**")
            
            df_stat = pd.DataFrame({
                "Min": np.round(classes_data[name]["min"], 4),
                "Mean (Центр)": np.round(classes_data[name]["mean"], 4),
                "Max": np.round(classes_data[name]["max"], 4)
            })
            st.dataframe(df_stat.T, use_container_width=True)
        else:
            ready = False
            st.warning("Завантажте хоча б один файл")

st.divider()

st.subheader("2. Розпізнавання невідомого образу")
test_file = st.file_uploader("Тестове зображення", type=["bmp", "png", "jpg", "jpeg"])

if test_file is not None and ready:
    t_abs, t_norm, t_grid, t_tot = get_features(test_file, cols, rows, threshold)

    st.image(t_grid, caption="Невідомий образ", width=220)
    st.write(f"**Абсолютний вектор:** {list(t_abs.astype(int))}")
    st.write(f"**Нормований вектор:** {[round(x, 4) for x in t_norm]}")

    st.divider()
    st.subheader("3. Результат класифікації")

    if "1)" in method:
        res = []
        for cname, cdata in classes_data.items():
            dist = float(np.sqrt(np.sum((t_norm - cdata["mean"]) ** 2)))
            res.append({"Клас": cname, "Відстань до центру": round(dist, 5)})

        df_res = pd.DataFrame(res).sort_values(by="Відстань до центру")
        st.dataframe(df_res, use_container_width=True)

        best = df_res.iloc[0]
        st.success(f"**Рішення:** Образ віднесено до **{best['Клас']}** (відстань = {best['Відстань до центру']})")

    else:
        res = []
        matched = []

        for cname, cdata in classes_data.items():
            inside = (t_norm >= cdata["min"]) & (t_norm <= cdata["max"])
            is_all = bool(np.all(inside))
            count = int(np.sum(inside))

            res.append({
                "Клас": cname,
                "Потрапив повністю?": "Так" if is_all else "Ні",
                "Співпало ознак": f"{count} / {len(t_norm)}"
            })
            if is_all:
                matched.append(cname)

        df_res = pd.DataFrame(res)
        st.dataframe(df_res, use_container_width=True)

        if len(matched) == 1:
            st.success(f"**Рішення:** Образ потрапив в область класу **{matched[0]}**")
        elif len(matched) > 1:
            st.warning(f"**Неоднозначність:** Образ потрапив одразу в декілька класів: {', '.join(matched)}")
        else:
            st.error("**Відмова:** Образ не потрапив у жодну область")


    report = f"Тестовий файл: {test_file.name}\nМетод: {method}\n"
    st.download_button("Завантажити звіт (.txt)", report, file_name="result_lab3.txt")

elif not ready:
    st.info("Завантажте всі зразки для навчання вище.")