# 1. IMPORTAÇÕES
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 2. CARREGANDO DATASET
iris = load_iris()
X = iris.data
y = iris.target

# Transformando em DataFrame só pra visualizar
df = pd.DataFrame(X, columns=iris.feature_names)
df['classe'] = y

print("Distribuição das classes:")
print(df['classe'].value_counts())
print("\n")

# 3. DIVISÃO DOS DADOS (estratificado!)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("Tamanho dos conjuntos:")
print(f"Treino: {len(X_train)}")
print(f"Validação: {len(X_val)}")
print(f"Teste: {len(X_test)}\n")

# 4. NORMALIZAÇÃO DOS DADOS (IMPORTANTE PARA MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 5. MODELO (Rede Neural MLP)
model = MLPClassifier(hidden_layer_sizes=(10, 10),
                      max_iter=1000,
                      random_state=42)

# 6. TREINAMENTO
model.fit(X_train, y_train)

# 7. VALIDAÇÃO
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"Acurácia na validação: {val_acc:.2f}\n")

# 8. TESTE FINAL
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Acurácia no teste: {test_acc:.2f}\n")

# 9. RELATÓRIO COMPLETO
print("Relatório de classificação:")
print(classification_report(y_test, y_test_pred))