import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Fungsi untuk melatih model
def train_model(data):
    reader = Reader(rating_scale=(data["rating"].min(), data["rating"].max()))
    surprise_data = Dataset.load_from_df(data[["user_id", "item_id", "rating"]], reader)
    trainset, testset = train_test_split(surprise_data, test_size=0.25)
    sim_options = {"name": "cosine", "user_based": True}  # Collaborative Filtering berbasis user
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return algo, rmse

# Fungsi untuk membuat prediksi
def make_prediction(model, user_id, item_id):
    pred = model.predict(uid=user_id, iid=item_id)
    return pred.est

# Streamlit App
st.title("Sistem Rekomendasi Collaborative Filtering")
st.write("Aplikasi untuk merekomendasikan film berdasarkan data IMDb.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset CSV", type="csv")

if uploaded_file:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset yang diunggah:")
    st.dataframe(df.head())

    # Validasi dataset
    if all(col in df.columns for col in ["Age Rating", "Title", "IMDb Rating"]):
        df = df.rename(columns={"Age Rating": "user_id", "Title": "item_id", "IMDb Rating": "rating"})
        st.write("Dataset valid. Memulai pelatihan model...")
        
        # Melatih model
        model, rmse = train_model(df)
        st.success(f"Model berhasil dilatih dengan RMSE: {rmse:.2f}")
        
        # Input untuk prediksi
        st.subheader("Buat Prediksi")
        user_id = st.text_input("Masukkan Age Rating (misalnya 'PG-13'):")
        item_id = st.text_input("Masukkan Judul Film (misalnya 'The Dark Knight'):")
        
        if st.button("Prediksi Rating"):
            if user_id and item_id:
                rating = make_prediction(model, user_id, item_id)
                st.write(f"Prediksi Rating untuk '{item_id}' dengan Age Rating '{user_id}': {rating:.2f}")
            else:
                st.warning("Harap masukkan Age Rating dan Judul Film!")
    else:
        st.error("Dataset harus memiliki kolom: 'Age Rating', 'Title', dan 'IMDb Rating'.")
