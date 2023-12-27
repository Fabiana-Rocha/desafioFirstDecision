class ObtemEmpresas:
    def __init__(self):
        pass

    def fazer_previsoes_completo(self, modelo, vetorizador, dados, coluna_texto='description',
                                 coluna_prevista='predicted_label'):
        """
        Faz previsões para o conjunto completo de dados usando um modelo treinado e um vetorizador.

        Parâmetros:
        - modelo: Modelo treinado.
        - vetorizador: Vetorizador treinado.
        - dados: DataFrame contendo os dados.
        - coluna_texto: Nome da coluna que contém o texto a ser previsto.
        - coluna_prevista: Nome da coluna onde as previsões serão armazenadas.

        Retorna:
        - DataFrame com previsões adicionadas na coluna especificada.
        """
        # Faz previsões para a coluna de texto
        previsoes = modelo.predict(vetorizador.transform(dados[coluna_texto]))

        # Adiciona as previsões ao DataFrame
        dados[coluna_prevista] = previsoes

        return dados

    def obtem_empresas_associadas(self, modelo, vetorizador, dados, coluna_texto='description',
                                  coluna_prevista='predicted_label', coluna_nome='name'):
        """
        Faz previsões para o conjunto completo de dados usando um modelo treinado e um vetorizador.
        Imprime os nomes das empresas associadas com base nas previsões.

        Parâmetros:
        - modelo: Modelo treinado.
        - vetorizador: Vetorizador treinado.
        - dados: DataFrame contendo os dados.
        - coluna_texto: Nome da coluna que contém o texto a ser previsto.
        - coluna_prevista: Nome da coluna onde as previsões serão armazenadas.
        - coluna_nome: Nome da coluna que contém os nomes das empresas.

        Retorna:
        - Nenhum (imprime os nomes das empresas associadas).
        """
        # Faz previsões para o conjunto completo de dados
        data_com_previsoes = self.fazer_previsoes_completo(modelo, vetorizador, dados, coluna_texto, coluna_prevista)

        # Filtra empresas associadas
        empresas_associadas = data_com_previsoes[data_com_previsoes[coluna_prevista] == True]

        # Obtém os nomes das empresas associadas
        empresas_associadas_nome = empresas_associadas[coluna_nome].tolist()

        # Imprime os nomes das empresas associadas
        print("Empresas Associadas:")
        for nome_empresa in empresas_associadas_nome:
            print(f"- {nome_empresa}")

            # Imprime a quantidade de empresas associadas
        quantidade_empresas_associadas = len(empresas_associadas_nome)
        print(f"\nTotal de Empresas Associadas: {quantidade_empresas_associadas}")

# In[ ]:




