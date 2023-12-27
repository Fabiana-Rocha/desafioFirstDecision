import openpyxl
import pandas as pd

class CarregaDados:
    def __init__(self, palavras_chave):
        self.palavras_chave = palavras_chave
        self.df = None

    def carrega_dados(self, path):
        opxl = openpyxl.load_workbook(path, data_only=True)
        sheet = opxl['canada_amostra']
        dados = [row for row in sheet.iter_rows(values_only=True)]
        self.df = pd.DataFrame(dados, columns=dados[0])
        return self.df

    def split_coluna(self):
        if self.df is None:
            raise ValueError("Os dados ainda não foram carregados. Utilize o método carrega_dados primeiro.")

        df_sep = self.df['name,description,employees,total_funding,city,subcountry,lat,lng'].str.split('"', expand=True)
        coluna_des = df_sep
        df_sep = df_sep.drop(columns=[1])
        df_sep2 = df_sep[2].str.split(',', expand=True)
        df_sep1 = df_sep[0].str.split(',', expand=True)
        name = df_sep1[0]
        employees = df_sep1[2]
        total_funding = df_sep1[3]
        city = df_sep1[4]
        subcountry = df_sep1[5]
        lat = df_sep1[6]
        lng = df_sep1[7]
        coluna_des[1] = coluna_des[1].fillna(df_sep1[1])
        description = coluna_des[1]
        data = pd.concat([name, description, employees, total_funding, city, subcountry, lat, lng], axis=1)
        data = data.rename(columns={0: 'name', 1: 'description', 2: 'employees', 3: 'total_funding', 4: 'city', 5: 'subcountry', 6: 'lat', 7: 'lng'})
        data.drop([0], inplace=True)
        data = data.dropna(subset=['description'])

        return data







