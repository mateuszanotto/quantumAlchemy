Existe duas formas de rodar o codigo:

Usando um unico script em python:
1. Rodar o useModel.py para
   1.1 Tirar todos dados dos .out e .hess
   1.2 Treinar o modelo com NablaChem
   1.3 Gerar as energias para todo espaço quimico (3 a 20 substituicoes) com carga maxima X
2. Analizar o out_{numero}.csv com 3dPlot.ipynb

OU

Usar os Jupyter-Notebooks (JN):
1. Rodar o getCSV.ipynb no jupyter-notebook para gerar o .csv
2. Rodar o Nablachem.ipynb para rodar o modelo
3. Rodar o 3dPlot.ipynb para analisar
