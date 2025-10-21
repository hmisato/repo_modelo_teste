# PREDIÇÃO

# Bibliotecas
import sys
import pandas as pd
from joblib import load

# Verificação dos argumentos de entrada
# O script deve receber dois argumentos: caminho do CSV de entrada e caminho do CSV de saída
if len(sys.argv) != 3:
    raise SystemExit('Uso: python predict.py <input.csv> <output.csv>')

caminho_entrada, caminho_saida = sys.argv[1], sys.argv[2]

# Carregamento do modelo treinado
modelo = load('modelos/modelo.pkl')

# Leitura dos dados para predição
X = pd.read_csv(caminho_entrada)

# Verificação se as colunas necessárias estão presentes
cols = ['IDADE',
        'VALOR_COMPRA_SITE',
        'QTDE_ITENS_COMPRA_SITE',
        'NOTA_SATISF_COMPRA_SITE',
        'QTDE_COMPRAS_LOJA_FISICA_6M',
        'QTDE_ITENS_LOJA_FISICA_6M',
        'FLAG_ACESSORIOS_COMPRA_SITE',
        'FLAG_FEMININA_COMPRA_SITE',
        'FLAG_MASCULINA_COMPRA_SITE',
        'FLAG_COMPROU_LOJA_FISICA_6M']
if not all(c in X.columns for c in cols):
    raise ValueError(f'O arquivo de entrada deve conter as seguintes colunas: {cols}')

# Realização das predições
pred = modelo.predict_proba(X[cols])[:, 1]

# Armazenamento das predições
pd.DataFrame({'prediction': pred}).to_csv(caminho_saida, index=False)
print(f'Predição concluída! Arquivo com resultados salvo em {caminho_saida}.')