import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Başlık ve açıklama
st.title("Ev Fiyat Tahmini Uygulaması")
st.write("Bu uygulama, evinizin özelliklerini girerek fiyat tahmini yapmanızı sağlar.")

# Model ve encoder'ı yükle
@st.cache_resource
def load_model():
    model = pickle.load(open("catboost_model.pkl", "rb"))
    encoders = pickle.load(open("label_encoders.pkl", "rb"))
    return model, encoders

model, encoders = load_model()

# Kategorik değişkenler
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea', 'furnishingstatus', 
                      "area_category", "ev_buyukluk_kategori"]

# Kullanıcı girişleri
st.header("Ev Özellikleri")

# Temel özellikler - Ana ekranda manuel giriş
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Alan (metrekare)", min_value=1000, max_value=15000, value=5000, step=100)
    bedrooms = st.number_input("Yatak Odası Sayısı", min_value=1, max_value=8, value=3, step=1)
    bathrooms = st.number_input("Banyo Sayısı", min_value=1, max_value=5, value=2, step=1)

with col2:
    stories = st.number_input("Kat Sayısı", min_value=1, max_value=4, value=2, step=1)
    parking = st.number_input("Park Yeri Sayısı", min_value=0, max_value=3, value=1, step=1)

# Kategorik özellikler - Yan menüde
st.sidebar.header("Diğer Özellikler")

mainroad = st.sidebar.selectbox("Ana Yol Üzerinde mi?", ["evet", "hayır"])
mainroad = 1 if mainroad == "evet" else 0

guestroom = st.sidebar.selectbox("Misafir Odası Var mı?", ["evet", "hayır"])
guestroom = 1 if guestroom == "evet" else 0

basement = st.sidebar.selectbox("Bodrum Katı Var mı?", ["evet", "hayır"])
basement = 1 if basement == "evet" else 0

hotwaterheating = st.sidebar.selectbox("Sıcak Su Isıtma Sistemi Var mı?", ["evet", "hayır"])
hotwaterheating = 1 if hotwaterheating == "evet" else 0

airconditioning = st.sidebar.selectbox("Klima Var mı?", ["evet", "hayır"])
airconditioning = 1 if airconditioning == "evet" else 0

prefarea = st.sidebar.selectbox("Tercih Edilen Bölgede mi?", ["evet", "hayır"])
prefarea = 1 if prefarea == "evet" else 0

furnishingstatus = st.sidebar.selectbox("Mobilya Durumu", ["mobilyalı", "yarı mobilyalı", "mobilyasız"])
if furnishingstatus == "mobilyalı":
    furnishingstatus = 2
elif furnishingstatus == "yarı mobilyalı":
    furnishingstatus = 1
else:
    furnishingstatus = 0

# Özellik mühendisliği
if st.button("Fiyat Tahmini Yap"):
    # Temel özellikleri içeren veri çerçevesi oluştur
    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }
    
    df_input = pd.DataFrame([input_data])
    
    # Özellik mühendisliği
    # 1. Oda başına alan
    df_input['area_per_bedroom'] = df_input['area'] / df_input['bedrooms']
    
    # 2. Banyo başına yatak odası
    df_input['bedroom_bathroom_ratio'] = df_input['bedrooms'] / df_input['bathrooms']
    
    # 3. Toplam oda sayısı
    df_input['total_rooms'] = df_input['bedrooms'] + df_input['bathrooms']
    
    # 4. Alan kategorisi
    df_input['area_category'] = pd.cut(df_input['area'], 
                                      bins=[0, 4000, 6000, float('inf')], 
                                      labels=[0, 1, 2])
    
    # 5. Lüks ev skoru
    df_input['luxury_score'] = (df_input['airconditioning'] + 
                               df_input['parking'] + 
                               df_input['prefarea'] + 
                               df_input['guestroom'])
    
    # 6. Kat başına oda sayısı
    df_input['rooms_per_story'] = df_input['total_rooms'] / df_input['stories']
    
    # 8. Büyük ev göstergesi
    df_input['is_large_house'] = (df_input['bedrooms'] >= 4).astype(int)
    
    # 9. Tüm imkanlar var mı?
    df_input['all_amenities'] = ((df_input['mainroad'] == 1) & 
                                (df_input['basement'] == 1) & 
                                (df_input['hotwaterheating'] == 1)).astype(int)
    
    # 10. Alan büyüklük skoru
    df_input['area_size_score'] = np.where(df_input['area'] < 3000, 1,
                                 np.where(df_input['area'] < 6000, 2, 3))
    
    # Toplam oda sayısı
    df_input['toplam_oda'] = df_input['bedrooms'] + df_input['bathrooms']
    
    # Oda başına alan
    df_input['oda_basina_alan'] = df_input['area'] / df_input['toplam_oda']
    
    # Kat başına oda sayısı
    df_input['kat_basina_oda'] = df_input['toplam_oda'] / df_input['stories']
    
    # Konfor skoru
    df_input['konfor_skoru'] = (df_input['mainroad'] + df_input['guestroom'] + 
                               df_input['basement'] + df_input['hotwaterheating'] + 
                               df_input['airconditioning'] + df_input['prefarea'])
    
    # Lüks ev göstergesi
    df_input['luks_ev'] = ((df_input['konfor_skoru'] >= 4) & 
                          (df_input['area'] > 5000)).astype(int)  # Medyan yerine sabit değer kullanıldı
    
    # Banyo/yatak odası oranı
    df_input['banyo_yatak_orani'] = df_input['bathrooms'] / df_input['bedrooms']
    
    # Park yeri başına alan
    df_input['park_yeri_basina_alan'] = df_input['area'] / (df_input['parking'] + 1)
    
    # Ev büyüklük kategorisi
    alan_esik_degerleri = [4000, 8000]  # Örnek eşik değerleri
    df_input['ev_buyukluk_kategori'] = pd.cut(df_input['area'], 
                                            bins=[0, alan_esik_degerleri[0], alan_esik_degerleri[1], float('inf')], 
                                            labels=['küçük', 'orta', 'büyük'])
    
    # Tüm özelliklerin mevcut olup olmadığı
    df_input['tum_ozellikler_var'] = ((df_input['mainroad'] == 1) & 
                                     (df_input['guestroom'] == 1) & 
                                     (df_input['basement'] == 1) & 
                                     (df_input['hotwaterheating'] == 1) & 
                                     (df_input['airconditioning'] == 1) & 
                                     (df_input['prefarea'] == 1)).astype(int)
    
    # Yaşam alanı kalitesi skoru
    df_input['yasam_alani_kalitesi'] = (df_input['area_size_score'] + 
                                       df_input['konfor_skoru'] + 
                                       df_input['stories'])
    
    # Banyo başına alan
    df_input['banyo_basina_alan'] = df_input['area'] / (df_input['bathrooms'] + 0.5)
    
    # Lüks banyo oranı
    df_input['luks_banyo_orani'] = (df_input['bathrooms'] >= 2).astype(int)
    
    # Premium konum skoru
    df_input['premium_konum_skoru'] = df_input['prefarea'] * 3 + df_input['mainroad'] * 2
    
    # Lüks konfor endeksi
    df_input['luks_konfor_endeksi'] = (df_input['airconditioning'] * 2 + 
                                      df_input['hotwaterheating'] * 3 + 
                                      df_input['basement'] * 1.5)
    
    # Mobilya puanı
    df_input['mobilya_puani'] = df_input['furnishingstatus']
    
    # Genel lükslük skoru
    df_input['genel_luksluk_skoru'] = (df_input['luks_konfor_endeksi'] + 
                                      df_input['premium_konum_skoru'] + 
                                      df_input['mobilya_puani'] * 2 + 
                                      df_input['stories'] + 
                                      df_input['parking'])
    
    # Basitlik skoru
    df_input['basitlik_skoru'] = 10 - (df_input['konfor_skoru'] / 2 + 
                                     df_input['mobilya_puani'] + 
                                     (df_input['stories'] > 1).astype(int) * 2)
    
    # Minimum konfor göstergesi
    df_input['minimum_konfor'] = ((df_input['bathrooms'] <= 1) & 
                                 (df_input['konfor_skoru'] <= 2) & 
                                 (df_input['mobilya_puani'] == 0)).astype(int)
    
    # Kategorik değişkenleri encode et
    for col in categorical_columns:
        if col in df_input.columns and col in encoders:
            df_input[col] = encoders[col].transform(df_input[col])
    
    # Tahmin yap
    prediction = model.predict(df_input)[0]
    
    # Sonucu göster
    st.header("Tahmin Sonucu")
    st.success(f"Tahmini Ev Fiyatı: {prediction:,.2f} TL")
    
    # Özellik önemini göster
    st.subheader("Girilen Ev Özellikleri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Temel Özellikler:**")
        st.write(f"Alan: {area} m²")
        st.write(f"Yatak Odası: {bedrooms}")
        st.write(f"Banyo: {bathrooms}")
        st.write(f"Kat Sayısı: {stories}")
        st.write(f"Park Yeri: {parking}")
    
    with col2:
        st.write("**Ek Özellikler:**")
        st.write(f"Ana Yol: {'Evet' if mainroad == 1 else 'Hayır'}")
        st.write(f"Misafir Odası: {'Var' if guestroom == 1 else 'Yok'}")
        st.write(f"Bodrum Katı: {'Var' if basement == 1 else 'Yok'}")
        st.write(f"Sıcak Su Isıtma: {'Var' if hotwaterheating == 1 else 'Yok'}")
        st.write(f"Klima: {'Var' if airconditioning == 1 else 'Yok'}")
        st.write(f"Tercih Edilen Bölge: {'Evet' if prefarea == 1 else 'Hayır'}")
        
        mobilya_durumu = "Mobilyalı" if furnishingstatus == 2 else "Yarı Mobilyalı" if furnishingstatus == 1 else "Mobilyasız"
        st.write(f"Mobilya Durumu: {mobilya_durumu}")
else:
    st.info("Fiyat tahmini için lütfen özellikleri ayarlayın ve 'Fiyat Tahmini Yap' butonuna tıklayın.")
