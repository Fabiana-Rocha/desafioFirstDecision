from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV

class TreinaModelo:
    def __init__(self):
        pass

    def treina_modelos(self, X_train_resampled, y_train_resampled, n_neighbors=5, random_state=42, cv=5, n_estimators=100):
        """
        Treina diferentes modelos usando os conjuntos resampleados.

        Parâmetros:
        - X_train_resampled: Conjunto de treinamento vetorizado após oversampling.
        - y_train_resampled: Rótulos após oversampling.
        - n_neighbors: Número de vizinhos para o KNeighborsClassifier, padrão é 5.
        - random_state: Semente para reprodução, padrão é 42.
        - cv: Número de dobras para a validação cruzada no LogisticRegressionCV, padrão é 5.
        - n_estimators: Número de estimadores para o GradientBoostingClassifier, padrão é 100.

        Retorna:
        - Lista contendo os modelos treinados.
        """
        modelos = []

        # Modelo 1: KNeighborsClassifier
        modelo_1 = KNeighborsClassifier(n_neighbors=n_neighbors)
        modelo_1.fit(X_train_resampled, y_train_resampled)
        modelos.append(modelo_1)

        # Modelo 2: RandomForestClassifier
        modelo_2 = RandomForestClassifier(random_state=random_state)
        modelo_2.fit(X_train_resampled, y_train_resampled)
        modelos.append(modelo_2)

        # Modelo 3: LogisticRegressionCV
        modelo_3 = LogisticRegressionCV(cv=cv, random_state=random_state)
        modelo_3.fit(X_train_resampled, y_train_resampled)
        modelos.append(modelo_3)

        # Modelo 4: GradientBoostingClassifier
        modelo_4 = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        modelo_4.fit(X_train_resampled, y_train_resampled)
        modelos.append(modelo_4)

        return modelos






