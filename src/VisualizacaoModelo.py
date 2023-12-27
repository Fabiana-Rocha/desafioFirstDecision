from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class VisualizacaoModelo:
    def __init__(self):
        pass

    def plota_matriz_confusao_e_distribuicao(self, modelo, vetorizador, dados, coluna_texto='description', coluna_rotulo='label'):
        """
        Gera a matriz de confusão e gráficos de distribuição das previsões para um modelo.

        Parâmetros:
        - modelo: Modelo treinado.
        - vetorizador: Vetorizador treinado.
        - dados: DataFrame contendo os dados.
        - coluna_texto: Nome da coluna que contém o texto a ser previsto.
        - coluna_rotulo: Nome da coluna que contém os rótulos reais.

        Retorna:
        - Nenhum (exibe os gráficos diretamente).
        """
        # Faz previsões para o conjunto de teste
        y_pred = modelo.predict(vetorizador.transform(dados[coluna_texto]))

        # Calcula a matriz de confusão
        conf_matrix = confusion_matrix(dados[coluna_rotulo], y_pred)

        # Plotar matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Não Tratamento de Água', 'Tratamento de Água'],
                    yticklabels=['Não Tratamento de Água', 'Tratamento de Água'])
        plt.xlabel('Previsão')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.show()

        # Gráfico de barras para mostrar a distribuição das previsões
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y_pred)
        plt.xticks([0, 1], ['Não Tratamento de Água', 'Tratamento de Água'])
        plt.xlabel('Previsão do Modelo')
        plt.ylabel('Contagem de Empresas')
        plt.title('Distribuição das Previsões do Modelo')
        plt.show()






