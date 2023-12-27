from CarregaDados import CarregaDados
from PreProcessor import PreProcessor
from TreinaModelo import TreinaModelo
from AvaliaModelo import AvaliaModelo
from SelecionaModelo import SelecionaModelo
from ObtemEmpresas import ObtemEmpresas
from VisualizacaoModelo import VisualizacaoModelo

class main:
        @staticmethod

        def run_workflow(path, palavras_chave):

                data_processor = CarregaDados(palavras_chave)
                text_preprocessor = PreProcessor(palavras_chave)
                model_trainer = TreinaModelo()
                model_evaluator = AvaliaModelo()
                model_selector = SelecionaModelo()
                prediction_analyzer = ObtemEmpresas()
                model_visualizer = VisualizacaoModelo()

                # Carregar dados
                dados = data_processor.carrega_dados(path)

                # Processar dados
                dados_processados = data_processor.split_coluna()

                # Pré-processar texto
                texto_preprocessado = dados_processados['description'].apply(text_preprocessor.preprocess_text)
                # texto_preprocessado = text_preprocessor.preprocess_text(dados_processados['description'])

                #cria rótulos para os dados
                dados_com_rotulos = text_preprocessor.cria_rotulos(dados_processados, 'description')
                # Dividir conjunto de dados
                X_train, X_test, y_train, y_test = text_preprocessor.divide_conjunto_dados(dados_processados, 'description', 'label')

                # Vetorização de texto usando TF-IDF
                X_train_vectorized, X_test_vectorized, vectorizer = text_preprocessor.vetoriza_texto_tfidf(X_train, X_test)

                # Aplica oversampling na classe minoritária
                X_train_resampled, y_train_resampled = text_preprocessor.aplica_oversampling(X_train_vectorized, y_train)

                # Treinar diferentes modelos usando os conjuntos resampleados
                modelos_treinados = model_trainer.treina_modelos(X_train_resampled, y_train_resampled)

                # Avaliar e selecionar o melhor modelo
                melhor_modelo = model_selector.avalia_e_seleciona_melhor_modelo(modelos_treinados, X_test, y_test, vectorizer)

                # Avaliar modelos no conjunto de teste original
                model_evaluator.avalia_modelos([melhor_modelo], X_test, y_test, vectorizer)

                # Avaliar modelos usando a métrica ROC-AUC e plota as curvas ROC
                model_evaluator.avalia_roc_auc([melhor_modelo], X_test, y_test, vectorizer)

                # Fazer previsões para o conjunto completo de dados
                data_com_previsoes = prediction_analyzer.fazer_previsoes_completo(melhor_modelo, vectorizer, dados_processados)

                # Imprimir empresas associadas
                prediction_analyzer.obtem_empresas_associadas(melhor_modelo, vectorizer, dados_processados)

                # Plotar matriz de confusão e gráficos de distribuição
                model_visualizer.plota_matriz_confusao_e_distribuicao(melhor_modelo, vectorizer, dados_processados)


if __name__ == "__main__":
    main.run_workflow('First_Desafio.xlsm', palavras_chave=['water treatment', 'solutions on waste and water', 'improve water quality and water efficiency use', 'water contamination', 'water for human consumption'])