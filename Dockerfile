FROM python:3

# define o diretório de trabalho
WORKDIR /app

#copia os arquivos do projeto para o container
COPY ./src /app

#instala as dependências
#COPY /src/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Defina o comando para iniciar sua aplicação
CMD ["python", "main.py"]