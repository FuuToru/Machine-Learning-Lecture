from flask import Flask, render_template, request
import numpy as np
from mnist import MNIST
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1,784)  
X_test = X_test.reshape(-1,784)

# Chuẩn hóa dữ liệu
X_train = X_train / 255.0
X_test = X_test / 255.0

# Xây dựng mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance')  # Chọn số láng giềng là 5, có thể điều chỉnh tùy ý
knn.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu ảnh từ request và chuyển đổi thành mảng numpy
    img_data = request.get_data()
    img_data = img_data.replace(b'data:image/png;base64,', b'')
    img_bytes = io.BytesIO(base64.b64decode(img_data))
    img = Image.open(img_bytes)

    # Thay đổi kích thước hình ảnh thành 28x28 pixels
    img_array = img.resize((28, 28))
    img_array.save('image.png')
    img_array = np.array(img_array)
    img_array = img_array.reshape(-1,784)  # Kích thước ảnh MNIST là 28x28=784

    # Chuẩn hóa dữ liệu ảnh
    img_array = img_array / 255.0

    
    # Dự đoán chữ số tay
    predicted_digit = knn.predict(img_array)[0]

    # Trả về kết quả dự đoán dưới dạng JSON
    return {'predicted_digit': int(predicted_digit)}

if __name__ == '__main__':
    app.run(debug=True)
