import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Membaca dataset
data = pd.read_csv('data_mahasiswa.csv')

# Memisahkan fitur dan target
X = data.drop('kelulusan', axis=1)
y = data['kelulusan']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat objek Naive Bayes
nb = GaussianNB()

# Melatih model Naive Bayes
nb.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = nb.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print('Akurasi:', accuracy)
