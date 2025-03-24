import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Đọc dữ liệu
df = pd.read_csv("E:/Du_an_ca_nhan/Cleaned_Sneakers_dataset.csv")

# Xử lý NaN trong Brand bằng giá trị xuất hiện nhiều nhất
most_common_brand = df["Brand"].mode()[0]  
df["Brand"].fillna(most_common_brand, inplace=True)

# Label Encoding
label_encoder = LabelEncoder()
df["Brand"] = label_encoder.fit_transform(df["Brand"])

# Lưu LabelEncoder & giá trị mode của Brand
joblib.dump(label_encoder, "E:/Du_an_ca_nhan/brand_encoder.pkl")
joblib.dump(most_common_brand, "E:/Du_an_ca_nhan/most_common_brand.pkl")

print(f"Đã lưu brand_encoder.pkl và most_common_brand.pkl (Mode: {most_common_brand})")
