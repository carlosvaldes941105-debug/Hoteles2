import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(page_title="Ocupación hotelera", layout="wide")
st.write(""" # Ocupación hotelera en los 70 destinos principales monitoreados​​​​​ """)
hotel_images = [
    ("hotel-reflect-krystal-grand-nuevo-vallarta-fachada-princ-min.jpg", "Krystal Grand Nuevo Vallarta"),
    ("454461679.jpg", "Catalonia Yucatan Beach"),
    ("hotel-dreams-tulum-fachada-princ-min.jpg", "Hotel Dreams Tulum"),
    ("642236603.jpg", "Fiesta Americana Acapulco Villas"),
]

image_columns = st.columns(2)
for image_index, (image_path, caption) in enumerate(hotel_images):
    with image_columns[image_index % 2]:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (900, 560), method=Image.Resampling.LANCZOS)
        st.image(image, caption=caption, use_container_width=True)

@st.cache_data
def load_hotel_data():
    hotel = pd.read_csv("Base70centros (1).csv", sep="\t", encoding="utf-8")
    hotel.columns = hotel.columns.str.strip().str.lower().str.replace(" ", "_")

    category_order = ["1 estrella", "2 estrellas", "3 estrellas", "4 estrellas", "5 estrellas", "Sin categoría"]
    category_map = {category: idx for idx, category in enumerate(category_order)}
    hotel["categoria_num"] = hotel["categoria"].astype(str).str.strip().map(category_map).fillna(len(category_order) - 1)

    destinos_list = sorted(hotel["centro"].dropna().astype(str).unique().tolist())
    centro_map = {destino: idx for idx, destino in enumerate(destinos_list)}
    hotel["centro_code"] = hotel["centro"].astype(str).map(centro_map)

    return hotel, category_order, destinos_list


st.header("Datos de evaluación")


def user_input_features(destinos_list, category_order):
    mes = st.number_input("Mes:", min_value=1, max_value=12, value=1, step=1)
    categoria = st.selectbox("Categoría:", options=category_order, index=0)
    destino = st.selectbox("Destino:", options=destinos_list, index=0)

    features = pd.DataFrame(
        {
            "mes": [mes],
            "categoria_num": [category_order.index(categoria)],
            "centro_code": [destinos_list.index(destino)],
        }
    )

    return features


hotel, category_order, destinos_list = load_hotel_data()
feature_columns = ["mes", "categoria_num", "centro_code"]

X1 = hotel[feature_columns]
Y1 = hotel["llegada_turistas_no_residentes"]

X2 = hotel[feature_columns]
Y2 = hotel["llegada_turistas_residentes"]

classifier1 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, max_features=7, random_state=0)
classifier1.fit(X1, Y1)

classifier2 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, max_features=7, random_state=0)
classifier2.fit(X2, Y2)

user_input = user_input_features(destinos_list, category_order)
prediction1 = classifier1.predict(user_input[feature_columns])
prediction2 = classifier2.predict(user_input[feature_columns])

st.subheader("Predicción de llegada de turistas")
st.write(f"Turistas no residentes: {int(round(prediction1[0])):,}".replace(",", "."))
st.write(f"Turistas residentes: {int(round(prediction2[0])):,}".replace(",", "."))
