from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class AvaliaModelo:
    def __init__(self):
        pass

    def avalia_modelos(self, modelos, X_test, y_test, vectorizer):
        """
        Avalia modelos em um conjunto de teste original.

        Parâmetros:
        - modelos: Lista contendo os modelos treinados.
        - X_test: Conjunto de teste com texto.
        - y_test: Rótulos do conjunto de teste.
        - vectorizer: Instância do vetorizador usado para treinamento.
        """
        for modelo in modelos:
            # Realiza previsões no conjunto de teste original
            y_pred = modelo.predict(vectorizer.transform(X_test))

            # Avalia o modelo
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Exibe as métricas de avaliação
            print(f'Modelo: {modelo.__class__.__name__}')
            print(f'Acurácia do modelo: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1-Score: {f1}')
            print()

    def avalia_roc_auc(self, modelos, X_test, y_test, vectorizer):
        """
        Avalia os modelos usando a métrica ROC-AUC e plota as curvas ROC.

        Parâmetros:
        - modelos: Lista contendo os modelos treinados.
        - X_test: Conjunto de teste com texto.
        - y_test: Rótulos do conjunto de teste.
        - vectorizer: Instância do vetorizador usado para treinamento.
        """
        for modelo in modelos:
            # Realiza previsões no conjunto de teste original
            y_pred = modelo.predict(vectorizer.transform(X_test))

            # Obtém a pontuação da classe positiva
            if hasattr(modelo, 'decision_function'):
                try:
                    y_pred_score = modelo.decision_function(vectorizer.transform(X_test))
                except AttributeError:
                    y_pred_score = modelo.predict_proba(vectorizer.transform(X_test))[:, 1]
            else:
                y_pred_score = y_pred

            # Avalia o modelo usando a métrica ROC-AUC
            roc_auc = roc_auc_score(y_test, y_pred_score)
            print(f'Modelo: {modelo.__class__.__name__}')
            print(f'ROC-AUC: {roc_auc}')

            # Plota a curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_score)
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('Taxa de Falsos Positivos (FPR)')
            plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
            plt.title('Curva ROC')
            plt.legend(loc='lower right')
            plt.show()
            print()







