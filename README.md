# 🛒 Yapay Zekâ Destekli Satış Tahmin Sistemi (XGBoost & LSTM)

Bu proje, perakende sektöründeki stok dengesizliğini (aşırı stok ve stoksuz kalma) önlemek amacıyla geliştirilmiş bir veri bilimi ve karar destek sistemidir. Modelleme süreçleri endüstri standardı olan CRISP-DM metodolojisi ile yürütülmüştür.

---

## 🚀 Proje Özeti
* **Hedef:** 1 milyondan fazla mağaza satış kaydını işleyerek; dışsal faktörlerin (tatil, promosyon vb.) etkisiyle günlük mağaza cirolarının tahmin edilmesi.
* **Özellik Mühendisliği (Feature Engineering):** Modelin zaman algısını güçlendirmek için gecikme özellikleri (Lag1, Lag7), hareketli ortalamalar (Roll7, Roll30) ve döngüsel zaman kodlaması (Sin/Cos) uygulanmıştır.
* **Modeller:** Karmaşık ve anlık dışsal şokları (kampanya, tatil) yakalamak için **XGBoost**; geçmiş zaman serisi trendlerini öğrenmek için **LSTM** (Sliding Window) kullanılmıştır. İki model, RMSE değerlerine dayalı ağırlıklı ortalama yöntemiyle (Stacking Ensemble) birleştirilmiştir.

---

## 📊 Model Başarı Metrikleri
* **XGBoost:** %95.89 R² (Açıklanabilirlik) | 753.88 RMSE
* **Hibrit Model (XGBoost + LSTM):** %93.70 R² (Açıklanabilirlik) | 933.13 RMSE

---

## 💻 Teknolojiler ve Kütüphaneler
* **Dil:** Python
* **Algoritmalar:** XGBoost, TensorFlow (Keras/LSTM), Scikit-Learn
* **Veri Analitiği & Görselleştirme:** Pandas, NumPy, Seaborn, Matplotlib
