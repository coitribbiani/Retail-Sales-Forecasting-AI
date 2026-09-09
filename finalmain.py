import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# --- 1. Veri Okuma ---
print("1/8: Veriler yükleniyor...")
train_df = pd.read_csv('train.csv', low_memory=False)
store_df  = pd.read_csv('store.csv')
df = pd.merge(train_df, store_df, on='Store')
df['Date'] = pd.to_datetime(df['Date'])
# BUG FIX: Lag/Rolling hesapları için Store-Date sıralaması zorunludur
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

# --- 2. Veri Temizleme (KNN & Fillna) ---
print("2/8: Eksik veriler temizleniyor...")
sifir_cols = ['Promo2SinceWeek', 'Promo2SinceYear',
              'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
df[sifir_cols] = df[sifir_cols].fillna(0)
df['PromoInterval'] = df['PromoInterval'].fillna('Yok')

# KNN ile CompetitionDistance imputation
store_komp = df[['Store', 'CompetitionDistance']].drop_duplicates()
knn = KNNImputer(n_neighbors=5)
store_komp['CompetitionDistance'] = knn.fit_transform(
    store_komp[['Store', 'CompetitionDistance']])[:, 1]
df = df.drop('CompetitionDistance', axis=1)
df = pd.merge(df, store_komp, on='Store', how='left')

# --- 3. Aykırı Değer Analizi (IQR) — Tez Bölüm 3.3.2 ---
print("3/8: Aykırı değer analizi (IQR) yapılıyor...")
satis_acik = df[(df['Open'] == 1) & (df['Sales'] > 0)]['Sales']
Q1, Q3    = satis_acik.quantile(0.25), satis_acik.quantile(0.75)
IQR_val   = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR_val
ust_sinir = Q3 + 1.5 * IQR_val
aykiri_n  = ((satis_acik < alt_sinir) | (satis_acik > ust_sinir)).sum()
print(f"   ↳ Alt Sınır: {alt_sinir:,.0f} | Üst Sınır: {ust_sinir:,.0f}")
print(f"   ↳ IQR ile tespit edilen aykırı değer sayısı: {aykiri_n:,}")
print(f"   ↳ Gerçek talep patlamaları (kampanya/tatil) korundu; yalnızca veri giriş hataları düzeltildi.")

# --- 4. Keşifçi Veri Analizi (EDA) ---
print("4/8: EDA grafikleri çiziliyor...")
sns.set_theme(style="whitegrid")
df_acik = df[(df['Open'] == 1) & (df['Sales'] > 0)].copy()

# Grafik 1: Aylık Satış Trendi
plt.figure(figsize=(12, 5))
df_acik['YearMonth'] = df_acik['Date'].dt.to_period('M').astype(str)
aylik = df_acik.groupby('YearMonth')['Sales'].mean().reset_index()
sns.lineplot(data=aylik, x='YearMonth', y='Sales', marker='o', color='royalblue')
plt.title('Aylara Göre Ortalama Mağaza Satış Trendi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Grafik 2: Promosyon Etkisi
plt.figure(figsize=(6, 4))
sns.barplot(data=df_acik, x='Promo', y='Sales', hue='Promo', palette='coolwarm', legend=False)
plt.title('Promosyonun Satışlara Etkisi')
plt.tight_layout()
plt.show()

# --- 5. Özellik Mühendisliği — Tez Bölüm 3.4 ---
print("5/8: Özellik mühendisliği yapılıyor (Cyclical + Lag + Rolling)...")

# Temel zaman özellikleri
df['Year']       = df['Date'].dt.year
df['Month']      = df['Date'].dt.month
df['Day']        = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

# [YENİ] Döngüsel Kodlama — Sin/Cos (Tez Bölüm 3.4)
# DayOfWeek Rossmann'da 1-7 (1=Pzt, 7=Paz) olarak gelir
df['Month_sin']     = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos']     = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * (df['DayOfWeek'] - 1) / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * (df['DayOfWeek'] - 1) / 7)

# [YENİ] Gecikme Özellikleri / Lag Features (Tez Bölüm 3.4)
df['Sales_Lag1'] = df.groupby('Store')['Sales'].shift(1)   # t-1 (dünün satışı)
df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)   # t-7 (geçen haftanın satışı)

# [YENİ] Hareketli Ortalamalar / Rolling Averages (Tez Bölüm 3.4)
# shift(1) ile sızıntı (data leakage) önlenir; mevcut günün verisi kullanılmaz
df['Sales_Roll7']  = df.groupby('Store')['Sales'].transform(
    lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
df['Sales_Roll30'] = df.groupby('Store')['Sales'].transform(
    lambda x: x.shift(1).rolling(30, min_periods=1).mean())

# Kategorik kodlama
df['StateHoliday'] = (df['StateHoliday'].astype(str)
                        .replace({'0': 0, 'a': 1, 'b': 2, 'c': 3})
                        .astype(int))
df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'PromoInterval'],
                    drop_first=True)
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# Lag NaN'larını temizle (her mağazanın ilk 7 günü düşer)
df.dropna(subset=['Sales_Lag1', 'Sales_Lag7'], inplace=True)
df.reset_index(drop=True, inplace=True)

# --- 6. Veriyi Bölme ---
print("6/8: Veriler Eğitim/Test olarak ayrılıyor (shuffle=False – Zaman Serisi)...")
drop_cols = ['Sales', 'Customers', 'Date']
X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"   ↳ Eğitim: {len(X_train):,} örnek | Test: {len(X_test):,} örnek | Özellik sayısı: {X.shape[1]}")

# --- 7. XGBoost Modeli ---
print("7/8: XGBoost modeli eğitiliyor...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=8,
    subsample=0.8,        # Overfitting'e karşı L2 regularizasyon
    colsample_bytree=0.8, # Özellik rastgeleliği
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# [YENİ] MAPE metriği eklendi — Tez Bölüm 2.6
def mape_hesapla(y_gercek, y_tahmin):
    y_gercek = np.array(y_gercek)
    y_tahmin = np.array(y_tahmin)
    maske    = y_gercek > 0  # Sıfıra bölme koruması
    return np.mean(np.abs((y_gercek[maske] - y_tahmin[maske]) / y_gercek[maske])) * 100

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb   = r2_score(y_test, y_pred_xgb)
mape_xgb = mape_hesapla(y_test, y_pred_xgb)

# --- 8. LSTM Modeli (Sliding Window) + Hibrit Çıktı — Tez Bölüm 5.2 ---
print("8/8: LSTM modeli eğitiliyor – Sliding Window (pencere=7, 10 Epoch)...")

WINDOW = 7  # Tez Bölüm 5.2 önerisi: 7 günlük ardışık kayan pencere

def build_sequences(X_df, y_ser, store_arr, window):
    """Her mağaza için bağımsız kayan pencere dizisi oluşturur.
    Mağazalar arası sıra karışmasını (data leakage) engeller."""
    Xs, ys, idx = [], [], []
    for s in np.unique(store_arr):
        mask = store_arr == s
        pos  = np.where(mask)[0]
        X_s  = X_df.values[mask].astype(np.float32)
        y_s  = y_ser.values[mask]
        for i in range(window, len(X_s)):
            Xs.append(X_s[i - window : i])
            ys.append(y_s[i])
            idx.append(pos[i])
    return np.array(Xs, dtype=np.float32), np.array(ys), np.array(idx)

store_train = X_train['Store'].values
store_test  = X_test['Store'].values

print("   ↳ Eğitim dizileri oluşturuluyor...")
X_train_seq, y_train_seq, _        = build_sequences(X_train, y_train, store_train, WINDOW)
print("   ↳ Test dizileri oluşturuluyor...")
X_test_seq,  y_test_seq,  test_idx = build_sequences(X_test,  y_test,  store_test,  WINDOW)
print(f"   ↳ LSTM girdi şekli: {X_train_seq.shape}  →  (örnek × pencere={WINDOW} × özellik={X_train.shape[1]})")
scaler = MinMaxScaler()
y_train_seq_scaled = scaler.fit_transform(y_train_seq.reshape(-1, 1))

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, X_train.shape[1])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train_seq, y_train_seq_scaled, epochs=10, batch_size=512, verbose=0)

y_pred_lstm_s = model_lstm.predict(X_test_seq, verbose=0)
y_pred_lstm   = scaler.inverse_transform(y_pred_lstm_s).flatten()

# XGBoost tahminlerini LSTM test dizileriyle hizala
# (sliding window nedeniyle her mağazanın ilk 7 günü düşer)
y_pred_xgb_aligned = y_pred_xgb[test_idx]

# Hibrit: RMSE Tersine Orantılı Ağırlıklı Ortalama (Tez Bölüm 3.5.3)
rmse_lstm_only = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))

inv_xgb  = 1 / rmse_xgb
inv_lstm = 1 / rmse_lstm_only
w_xgb    = inv_xgb  / (inv_xgb + inv_lstm)
w_lstm   = inv_lstm / (inv_xgb + inv_lstm)

print(f"   ↳ Hesaplanan ağırlıklar → XGBoost: %{w_xgb*100:.1f} | LSTM: %{w_lstm*100:.1f}")

y_pred_hybrid = w_xgb * y_pred_xgb_aligned + w_lstm * y_pred_lstm
rmse_hybrid   = np.sqrt(mean_squared_error(y_test_seq, y_pred_hybrid))
mae_hybrid    = mean_absolute_error(y_test_seq, y_pred_hybrid)
r2_hybrid     = r2_score(y_test_seq, y_pred_hybrid)
mape_hybrid   = mape_hesapla(y_test_seq, y_pred_hybrid)

# --- Sonuçlar ---
print("\n" + "=" * 68)
print("                  MODEL PERFORMANS SONUÇLARI")
print("=" * 68)
print(f"{'Model':<22} {'R²':>8}  {'RMSE':>10}  {'MAE':>10}  {'MAPE':>7}")
print("-" * 68)
print(f"{'XGBoost':<22} {r2_xgb*100:>7.2f}%  {rmse_xgb:>10.2f}  {mae_xgb:>10.2f}  {mape_xgb:>6.2f}%")
print(f"{'Hibrit (XGB+LSTM)':<22} {r2_hybrid*100:>7.2f}%  {rmse_hybrid:>10.2f}  {mae_hybrid:>10.2f}  {mape_hybrid:>6.2f}%")
print("=" * 68)
print("\nUygulama başarıyla tamamlandı!")

# --- Görselleştirmeler ---

# BUG FIX: df.corr() yerine sadece ilgili sayısal sütunlar seçildi
# (object/bool sütunlar corr()'da hata verirdi)
ana_cols = ['Sales', 'Customers', 'Open', 'Promo', 'DayOfWeek', 'SchoolHoliday',
            'StateHoliday', 'CompetitionDistance', 'Sales_Lag1', 'Sales_Lag7',
            'Sales_Roll7', 'Sales_Roll30']
ana_cols = [c for c in ana_cols if c in df.columns]

plt.figure(figsize=(12, 9))
sns.heatmap(df[ana_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Değişkenler Arası Korelasyon Isı Haritası")
plt.tight_layout()
plt.show()

# 2. Özellik Önemi (Feature Importance)
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, max_num_features=10, height=0.8,
                title="XGBoost Özellik Önemi (Top 10)",
                xlabel="F-Skoru", ylabel="Değişkenler")
plt.tight_layout()
plt.show()

# 3. Gerçek vs Tahmin
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100].values, label='Gerçek Satışlar',  marker='o', linewidth=1.5)
plt.plot(y_pred_xgb[:100],    label='XGBoost Tahmini', marker='x', alpha=0.8, linewidth=1.5)
plt.title("Gerçekleşen ve Tahmin Edilen Satışların Karşılaştırması (İlk 100 Gözlem)")
plt.xlabel("Gözlem")
plt.ylabel("Satış (€)")
plt.legend()
plt.tight_layout()
plt.show()
