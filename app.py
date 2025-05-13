import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel

# Thiết lập trang
st.set_page_config(
    page_title="Dự Đoán Giá Nhà",
    page_icon="🏠",
    layout="wide"
)

# Khởi tạo Spark session
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("HousePricePrediction") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

# Hàm tải thông tin mô hình
@st.cache_resource
def load_model_info():
    try:
        model_path = "models/house_price_model.pkl"
        with open(model_path, "rb") as f:
            model_info = pickle.load(f)
        return model_info
    except Exception as e:
        st.error(f"Lỗi khi tải thông tin mô hình: {e}")
        return None
    
# Hàm dự đoán giá sử dụng hệ số trực tiếp
def predict_price(bhk, total_sqft, bath, balcony):
    try:
        # Tải thông tin mô hình
        model_info = load_model_info()
        
        if model_info is None:
            st.warning("Không tìm thấy thông tin mô hình, sử dụng giá trị mẫu")
            # Giá trị mẫu - thay thế bằng hệ số thực tế nếu biết
            coefficients = [5.0, 0.1, 15.0, 3.0]
            intercept = 10.0
        else:
            coefficients = model_info["coefficients"]
            intercept = model_info["intercept"]
        
        # Tạo vector đặc trưng
        features = [float(bhk), float(total_sqft), float(bath), float(balcony)]
        
        # Tính toán dự đoán
        prediction = sum(coef * feat for coef, feat in zip(coefficients, features)) + intercept
        
        return prediction
    
    except Exception as e:
        st.error(f"Lỗi trong quá trình dự đoán: {e}")
        return None

# Hiển thị thông tin mô hình
def show_model_info():
    model_info = load_model_info()
    if model_info:
        st.write("**Thông số mô hình:**")
        st.write(f"- Số lần lặp tối đa: {model_info.get('max_iter', 'Không có thông tin')}")
        st.write(f"- Tham số điều chỉnh: {model_info.get('reg_param', 'Không có thông tin')}")
        st.write(f"- Tham số elastic net: {model_info.get('elastic_net_param', 'Không có thông tin')}")
        
        # Hiển thị hệ số của các đặc trưng
        features = model_info.get("features", ["BHK", "total_sqft", "bath", "balcony"])
        coefficients = model_info.get("coefficients", [0, 0, 0, 0])
        
        coef_data = {
            "Đặc trưng": features,
            "Hệ số": coefficients
        }
        
        st.write("**Hệ số của các đặc trưng:**")
        st.dataframe(pd.DataFrame(coef_data))
        
        st.write(f"**Hằng số (Intercept):** {model_info.get('intercept', 'Không có thông tin')}")
    else:
        st.warning("Không thể tải thông tin mô hình")

# Thiết lập giao diện người dùng
st.title("🏠 Ứng Dụng Dự Đoán Giá Nhà")
st.write("Nhập thông tin căn hộ để nhận dự đoán giá")

# Tạo tabs
tab1, tab2 = st.tabs(["Nhập liệu", "Thông tin mô hình"])

with tab1:
    # Tạo layout cột
    col1, col2 = st.columns(2)
    
    with col1:
        # Các trường đầu vào
        bhk = st.number_input("Số phòng ngủ (BHK)", 
                             min_value=1, 
                             max_value=5, 
                             value=2,
                             help="Số phòng ngủ của căn hộ")
        
        total_sqft = st.number_input("Diện tích (sqft)", 
                                    min_value=300.0, 
                                    max_value=10000.0, 
                                    value=1000.0, 
                                    step=50.0,
                                    help="Tổng diện tích căn hộ tính bằng feet vuông")
    
    with col2:
        bath = st.number_input("Số phòng tắm", 
                              min_value=1, 
                              max_value=5, 
                              value=2,
                              help="Số phòng tắm trong căn hộ")
        
        balcony = st.number_input("Số ban công", 
                                 min_value=0, 
                                 max_value=4, 
                                 value=1,
                                 help="Số ban công của căn hộ")
    
    # Nút dự đoán
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Dự Đoán Giá", use_container_width=True)
    
    # Xử lý khi nhấn nút dự đoán
    if predict_button:
        with st.spinner("Đang tính toán..."):
            price = predict_price(bhk, total_sqft, bath, balcony)
            
            if price is not None:
                # Hiển thị kết quả
                st.success(f"Giá dự đoán: {price:,.2f} (đơn vị: triệu)")
                
                # Hiển thị thông tin chi tiết
                st.subheader("Chi tiết căn hộ")
                details = {
                    "Đặc điểm": ["Số phòng ngủ (BHK)", "Diện tích (sqft)", "Số phòng tắm", "Số ban công"],
                    "Giá trị": [bhk, total_sqft, bath, balcony]
                }
                st.dataframe(pd.DataFrame(details))
                
                # Biểu đồ tầm quan trọng của các đặc trưng
                st.subheader("Tầm quan trọng của các đặc trưng")
                
                # Lấy hệ số từ mô hình
                model_info = load_model_info()
                if model_info is not None and "coefficients" in model_info:
                    importances = [abs(x) for x in model_info["coefficients"]]
                    features = model_info.get("features", ["BHK", "Diện tích", "Phòng tắm", "Ban công"])
                else:
                    # Giá trị mẫu nếu không tìm thấy thông tin
                    importances = [5.0, 0.1, 15.0, 3.0]
                    features = ["BHK", "Diện tích", "Phòng tắm", "Ban công"]
                
                # Tạo DataFrame cho biểu đồ
                importance_df = pd.DataFrame({
                    'Đặc trưng': features,
                    'Mức độ ảnh hưởng': importances
                })
                
                # Hiển thị biểu đồ
                st.bar_chart(importance_df.set_index('Đặc trưng'))
            else:
                st.error("Không thể dự đoán giá. Vui lòng kiểm tra lại giá trị đầu vào.")

with tab2:
    st.subheader("Thông tin về mô hình")
    show_model_info()
    
    # Hiển thị mẫu dữ liệu
    st.subheader("Mẫu dữ liệu huấn luyện")
    
    # Tạo dữ liệu mẫu - thay thế bằng dữ liệu thật nếu có thể
    sample_data = {
        "BHK": [1, 1, 1, 2, 3, 4],
        "total_sqft": [435.0, 550.0, 440.0, 1077.0, 2065.0, 2825.0],
        "bath": [1, 1, 1, 2, 4, 4],
        "balcony": [1, 1, 0, 2, 1, 3],
        "price": [19.5, 27.0, 28.0, 93.0, 210.0, 250.0]
    }
    
    st.dataframe(pd.DataFrame(sample_data))

# Thêm thông tin sidebar
st.sidebar.title("Giới thiệu")
st.sidebar.info("""
Ứng dụng này dự đoán giá nhà dựa trên các tham số như số phòng ngủ (BHK), 
diện tích tổng (sqft), số phòng tắm và số ban công.

Dự đoán được thực hiện dựa trên mô hình Hồi quy tuyến tính
với các hệ số được lưu trữ từ quá trình huấn luyện.
""")

# Thêm hướng dẫn sử dụng
st.sidebar.title("Hướng dẫn sử dụng")
st.sidebar.write("""
1. Nhập các thông số của căn hộ vào biểu mẫu
2. Nhấn nút "Dự Đoán Giá"
3. Kết quả sẽ hiển thị cùng với biểu đồ phân tích
""")

# Thêm thông tin liên hệ
st.sidebar.title("Liên hệ")
st.sidebar.write("Email: email@example.com")