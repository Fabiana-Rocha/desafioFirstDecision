from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class SelecionaModelo:
    def __init__(self):
        pass

    def avalia_e_seleciona_melhor_modelo(self, modelos, X_test, y_test, vectorizer, pesos=(1, 1, 1)):
        """
        Avalia modelos com base em várias métricas e retorna o modelo com a média ponderada mais alta.

        Parâmetros:
        - modelos: Lista contendo os modelos treinados.
        - X_test: Conjunto de teste com texto.
        - y_test: Rótulos do conjunto de teste.
        - vectorizer: Instância do vetorizador usado para treinamento.
        - pesos: Pesos para cada métrica (accuracy, F1-score, ROC-AUC).

        Retorna:
        - Melhor modelo com base na média ponderada das métricas.
        """
        melhor_modelo = None
        melhor_pontuacao = 0.0

        for modelo in modelos:
            # Realiza previsões no conjunto de teste original
            y_pred = modelo.predict(vectorizer.transform(X_test))

            # Calcula métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Obtém a pontuação da classe positiva para o ROC-AUC
            if hasattr(modelo, 'decision_function'):
                try:
                    y_pred_score = modelo.decision_function(vectorizer.transform(X_test))
                except AttributeError:
                    y_pred_score = modelo.predict_proba(vectorizer.transform(X_test))[:, 1]
            else:
                y_pred_score = y_pred

            roc_auc = roc_auc_score(y_test, y_pred_score)

            # Calcula a média ponderada das métricas
            pontuacao_media_ponderada = (pesos[0] * accuracy + pesos[1] * f1 + pesos[2] * roc_auc) / sum(pesos)

            # Atualiza o melhor modelo, se necessário
            if pontuacao_media_ponderada > melhor_pontuacao:
                melhor_modelo = modelo
                melhor_pontuacao = pontuacao_media_ponderada

        return melhor_modelo






