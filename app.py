import streamlit as st
import joblib
import pandas as pd
import os

print(f"File tồn tại: {os.path.exists('shoe_price_model.pkl')}")
print(f"Quyền truy cập file: {os.access('shoe_price_model.pkl', os.R_OK)}")

# Load model & encoder
model = joblib.load("shoe_price_model.pkl")
label_encoder = joblib.load("brand_encoder.pkl")
most_common_brand = joblib.load("most_common_brand.pkl")  # Load giá trị mode

# Giao diện web
st.title("Dự đoán phân khúc giá giày")

# Nhập thông tin
brand_list = list(label_encoder.classes_)  # Lấy danh sách thương hiệu đã mã hóa
brand = st.selectbox("Chọn thương hiệu", brand_list)
count_of_rating = st.number_input("Số lượng đánh giá", min_value=0, step=1)
rating = st.slider("Điểm đánh giá", 0.0, 5.0, 4.0)
price = st.number_input("Nhập giá gốc (USD)", min_value=0, step=10000)
discount = st.slider("Giảm giá (%)", 0.0, 100.0, 10.0)

# Dự đoán phân khúc
if st.button("Dự đoán phân khúc"):
    # Xử lý trường hợp Brand bị NaN
    if brand == "":
        brand = most_common_brand  # Thay thế bằng giá trị mode đã lưu trước đó

    # Chuyển đổi Brand từ tên → số
    brand_encoded = label_encoder.transform([brand])[0]

    # Chuẩn bị dữ liệu đầu vào
    data = pd.DataFrame([[brand_encoded, count_of_rating, rating, price, discount]], 
                        columns=["Brand", "Count_of_Rating", "Rating", "Price", "Discount"])
    
    # Dự đoán phân khúc
    predicted_segment = model.predict(data)[0]

    # Hiển thị kết quả
    st.success(f"Phân khúc dự đoán: {predicted_segment} ({brand})")
