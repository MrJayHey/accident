import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

class AccidentRiskModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.out(x)
input_dim = 12
device = torch.device("cpu")

model = AccidentRiskModel(input_dim)
model.load_state_dict(torch.load("accident_risk_model.pth", map_location=device))
model.eval()

scaler = joblib.load("scaler.pkl")

# ---- Интерфейс пользователя ----
st.title("Оценка вероятности аварии водителя")
КБМ = st.number_input("КБМ", min_value=0.0, max_value=3.0, value=0.46, step=0.01)
Прицеп = st.selectbox("Есть прицеп?", ["Нет", "Да"])
ПопулярноеИмя = st.text_input('Имя водителя','Кирилл')
КрупныйНасПункт = st.selectbox("В нас. пункте проживания более 30 тысяч жителей?", ["Нет", "Да"])
Стаж_водителя = st.number_input("Стаж водителя (лет)", min_value=0, max_value=80, value=15)
МощностьДвигателя = st.number_input("Мощность двигателя (л.с.)", min_value=10, max_value=1000, value=150)
VIN = st.selectbox("Есть VIN номер?", ["Нет", "Да"])
Мужчина = st.selectbox("Пол", ["Женщина", "Мужчина"])
ВозрастСобственника = st.number_input("Возраст водителя", min_value=18, max_value=100, value=18)
ВозрастТС = st.number_input("Возраст ТС", min_value=0, max_value=100, value=10)
ОтечественнаяМарка = st.selectbox("Отечественная марка ТС?", ["Нет", "Да"])
Частота = st.text_input("Номер ТС")
chast = pd.read_excel('Частота.xlsx', dtype={'Код':str})
# ---- Кнопка предсказания ----
if st.button("Оценить риск"):
    ПопулярноеИмя = ПопулярноеИмя.lower() in ['кирилл', 'алексей']
    Частота = Частота[-2:]
    Частота = chast[chast.Код==Частота]['Частота']
    # Преобразование данных
    input_data = pd.DataFrame([{
        'КБМ': КБМ,
        'Прицеп': 1.0 if Прицеп == "Да" else 0.0,
        'ПопулярноеИмя': ПопулярноеИмя,
        'Крупный нас. пункт?': 1 if КрупныйНасПункт == "Да" else 0,
        'Стаж_водителя': Стаж_водителя,
        'МощностьДвигателя': МощностьДвигателя,
        'VIN': 1 if VIN == "Да" else 0,
        'Мужчина': 1 if Мужчина == "Мужчина" else 0,
        'ВозрастСобственника': ВозрастСобственника,
        'ВозрастТС': ВозрастТС,
        'ОтечественнаяМарка': 1 if ОтечественнаяМарка == "Да" else 0,
        'Частота': Частота
    }])
    X_scaled = scaler.transform(input_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(X_tensor)
        probability = torch.sigmoid(output).item()

    # Отображение результата
    if probability<=0.3:
        st.success(f"Низкая аварийность.")
    elif probability<=0.6:
        st.info(f"Умеренная аварийность.")
    elif probability<=0.8:
        st.warning(f"Высокая аварийность.")
    else:
        st.error(f"Критическая аварийность.")
