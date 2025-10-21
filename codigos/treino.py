# TREINO DO MODELO

# Bibliotecas
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
import os

# Caminho do arquivo de dados de treino
caminho = os.path.join('dados', 'entrada', 'construcao', 'dados_construcao_2025_06.csv')

# Leitura de dados de treino
dados = pd.read_csv(caminho)

# Definição das variáveis explicativas e resposta
X = dados[['IDADE',
           'VALOR_COMPRA_SITE',
           'QTDE_ITENS_COMPRA_SITE',
           'NOTA_SATISF_COMPRA_SITE',
           'QTDE_COMPRAS_LOJA_FISICA_6M',
           'QTDE_ITENS_LOJA_FISICA_6M',
           'FLAG_ACESSORIOS_COMPRA_SITE',
           'FLAG_FEMININA_COMPRA_SITE',
           'FLAG_MASCULINA_COMPRA_SITE',
           'FLAG_COMPROU_LOJA_FISICA_6M']]
y = dados['FLAG_RECOMPRA_PROX_3_MESES']

# Treino do modelo de regressão logística múltipla
modelo = LogisticRegression(max_iter=1000)

modelo.fit(X, y)

# Armazenamento do modelo treinado
os.makedirs('modelos', exist_ok=True)
dump(modelo, 'modelos/modelo.pkl')
print('Modelo treinado! O arquivo modelo.pkl foi gerado e salvo em \\modelos.')