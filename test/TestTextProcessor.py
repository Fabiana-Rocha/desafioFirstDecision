import unittest
import pandas as pd

from src.PreProcessor import PreProcessor

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        # Configuração inicial dos dados de teste
        self.palavras_chave = ['water', 'treatment', 'quality']
        self.text_processor = PreProcessor(palavras_chave=self.palavras_chave)

    def test_preprocess_text(self):
        # Teste para o método preprocess_text

        # Texto de entrada
        input_text = "Water treatment solutions for improving water quality."

        # Saída esperada após o pré-processamento
        expected_output = "water treatment solutions improving water quality"

        # Aplica o pré-processamento
        processed_text = self.text_processor.preprocess_text(input_text)

        # Verifica se a saída é igual à saída esperada
        self.assertEqual(processed_text, expected_output)

    def test_cria_rotulos(self):
        # Teste para o método cria_rotulos

        # Dados de teste
        data = pd.DataFrame({
            'description': ['Water treatment solutions', 'Some other description', 'Water contamination issues']
        })

        # Aplica o método cria_rotulos
        data_with_labels = self.text_processor.cria_rotulos(data, 'description')

        # Verifica se a coluna 'label' foi criada e se os rótulos estão corretos
        self.assertIn('label', data_with_labels.columns)
        self.assertListEqual(list(data_with_labels['label']), [1, 0, 1])

if __name__ == '__main__':
    unittest.main()
