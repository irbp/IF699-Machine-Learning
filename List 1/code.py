
# coding: utf-8

# # Aprendizagem de Máquina - Relatório 1
# 
# 
# ## Questão 1
# 
# 
# ### Objetivos
# 
# Nesta questão, o objetivo é desenvolver um algoritmo que implemente o classificador k-NN com peso e sem peso. Logo após, o desempenho do classificador é avaliado utilizando dois datasets com atributos numéricos, onde será mostrado o comportamento para diferentes valores de k.
# 
# ### Metodologia
# 
# Para a avaliação do algoritmo proposto pela questão, foram utilizados duas bases de dados obtidas no Promise Repository (http://promise.site.uottawa.ca/SERepository/datasets-page.html). O primeiro deles foi o "KC2/Software defect prediction" e o segundo foi o "JM1/Software defect prediction".
# As duas bases de dados estavam contidas cada uma em um arquivo ARFF (Atribute-Relation File Format), onde cada linha representa uma instância e as colunas representam os atributos juntamente com as classes.
# Para a leitura deste formato de arquivo foi utilizado o método arff da biblioteca Scipy para Python assim como a biblioteca Pandas que é voltada para a análise de dados. A biblioteca Matplotlib também foi utilizada para gerar os gráficos utilizados na análise.

# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')

from src.knn import *
import datetime
from scipy.io import arff
import matplotlib.pyplot as plt
import datetime

time = []


# Abaixo segue um resumo da primeira base de dados:

# In[2]:


data_1 = arff.loadarff('datasets/kc2.arff')
df_1 = pd.DataFrame(data_1[0])
print("KC2/Software defect prediction")
df_1.head()


# In[3]:


df_1.describe()


# Podemos notar um desbalanceamento das classes no gráfico exibido abaixo:

# In[4]:


df_1['problems'].value_counts().plot(kind='bar')


# Agora analisando a segunda base de dados, temos:

# In[51]:


data_2 = arff.loadarff('datasets/jm1.arff')
df_2 = pd.DataFrame(data_2[0])
print("JM1/Software defect prediction")
df_2.head()


# In[6]:


df_2.describe()


# Assim como a primeira base de dados, podemos observar aqui também uma desproporção entre as classes do problema:

# In[7]:


df_2['defects'].value_counts().plot(kind='bar')


# O algoritmo do k-NN foi implementado também em Python e se encontra do diretório "src/". Neste diretório encontram-se dois arquivos: distances.py e knn.py, onde o primeiro implementa as distâncias utilizadas nesta e nas próximas questões e o último é a implementação do classificador k-NN.

# ### Resultados
# 
# Para a avaliação dos classificadores k-NN com e sem peso, cada base de dados foi submetida ao k-fold cross validation, onde o valor escolhido para k foi 10. Cada classificador executou para os números de vizinhos k = {1,2,3,5,7,9,11,13,15}.
# As acurácias obtidas utilizando a primeira base de dados seguem abaixo:

# In[8]:


n_fold = 10
k_values = [1,2,3,5,7,9,11,13,15]

data_1 = df_1.values
np.random.shuffle(data_1)

mod = data_1.shape[0] % n_fold
data_1 = data_1[mod:]

accs = []
weighted_accs = []

print("---------------------KNN---------------------")
cur = datetime.datetime.now()
accs = cross_validation(data_1, n_fold, k_values)
time.append(datetime.datetime.now() - cur)
for i in range(len(k_values)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], accs[i]))

print("\n---------------------WEIGHTED KNN---------------------")
cur = datetime.datetime.now()
weighted_accs = cross_validation(data_1, n_fold, k_values, with_weight=True)
time.append(datetime.datetime.now() - cur)
for i in range(len(k_values)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], weighted_accs[i]))


# A comparação dos dois classificadores variando o número de vizinhos pode ser visualizada no gráfico abaixo:

# In[24]:


fig, ax = plt.subplots()

index = np.arange(len(k_values))
bar_width = 0.35
opacity = 0.4

rects1 = ax.bar(index, accs, bar_width,
                alpha=opacity, color='b',
                label='k-NN')

rects2 = ax.bar(index + bar_width, weighted_accs,
                bar_width, color='r',
                alpha=opacity, label='Weighted k-NN')

ax.set_xlabel('k Neighbors')
ax.set_ylabel('Accuracy (%)')
ax.set_title('k-NN Performance')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('1', '2', '3', '5', '7', '9', '11', '13', '15'))
ax.legend()

fig.tight_layout()
plt.show()


# Agora avaliando a segunda base de dados:

# In[ ]:


n_fold = 10
k_values = [1,2,3,5,7,9,11,13,15]

data_2 = df_2.values
np.random.shuffle(data_2)

mod = data_2.shape[0] % n_fold
data_2 = data_2[mod:]

accs = []
weighted_accs = []

print("---------------------KNN---------------------")
cur = datetime.datetime.now()
accs = cross_validation(data_2, n_fold, k_values)
time.append(datetime.datetime.now() - cur)
for i in range(len(k_values)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], accs[i]))

print("\n---------------------WEIGHTED KNN---------------------")
cur = datetime.datetime.now()
weighted_accs = cross_validation(data_2, n_fold, k_values, with_weight=True)
time.append(datetime.datetime.now() - cur)
for i in range(len(k_values)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], weighted_accs[i]))


# ## Questão 2
# 
# 
# ### Objetivos
# 
# Nesta questão, o objetivo é desenvolver um algoritmo que implemente o classificador k-NN com peso e sem peso utilizando a distância VDM . Logo após, o desempenho do classificador é avaliado utilizando dois datasets com atributos categóricos, onde será mostrado o comportamento para diferentes valores de k.
# 
# ### Metodologia
# 
# Para a avaliação do algoritmo proposto pela questão, foram utilizados duas bases de dados obtidas no UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets.html). O primeiro deles foi o "Chess (King-Rook vs. King-Pawn)" e o segundo foi o "Tic-Tac-Toe Endgame Data Set".
# As duas bases de dados estavam contidas cada uma em um arquivo .DATA, onde cada linha representa uma instância e as colunas representam os atributos juntamente com as classes.
# Para a leitura deste formato de arquivo foi utilizado a biblioteca Pandas. A biblioteca Matplotlib também foi utilizada para gerar os gráficos utilizados na análise.
# Abaixo segue um resumo da primeira base de dados:

# In[26]:


df_1 = pd.read_csv('datasets/chess.data')
pd.set_option('display.max_columns', df.shape[1])
df_1.head()


# In[27]:


df_1.describe()


# Podemos notar que as classes estão bem distribuídas através do gráfico abaixo:

# In[29]:


df_1['C'].value_counts().plot(kind='bar')


# Agora analisando a segunda base de dados, temos:

# In[53]:


df_2 = pd.read_csv('datasets/tic-tac-toe.data')
pd.set_option('display.max_columns', df_2.shape[1])
df_2.head()


# In[38]:


df_2.describe()


# Observando o gráfico abaixo pode-se notar um certo desbalanceamento entre as classes:

# In[39]:


df_2['C'].value_counts().plot(kind='bar')


# ### Resultados
# 
# Para a avaliação dos classificadores k-NN com e sem peso, cada base de dados foi submetida ao k-fold cross validation, onde o valor escolhido para k foi 10. Cada classificador executou para os números de vizinhos k = {1,2,3,5,7,9,11,13,15}.
# As acurácias obtidas utilizando a primeira base de dados seguem abaixo:

# In[30]:


n_fold = 10
data_1 = df_1.values
np.random.shuffle(data_1)
mod = data_1.shape[0] % n_fold
data_1 = data_1[mod:]

k_values = [1,2,3,5,7,9,11,13,15]
accs = []
weighted_accs = []

print("---------------------KNN---------------------")
cur = datetime.datetime.now()
accs = cross_validation(data_1, n_fold, k_values, data_type=1, dataframe=df)
time.append(datetime.datetime.now() - cur)
for i in range(len(accs)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], accs[i]))

print("---------------------WEIGHTED KNN---------------------")
cur = datetime.datetime.now()
accs = cross_validation(data_1, n_fold, k_values, data_type=1, dataframe=df, with_weight=True)
time.append(datetime.datetime.now() - cur)
for i in range(len(accs)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], accs[i]))


# Abaixo seguem os resultados para a segunda base de dados:

# In[54]:


data_2 = df_2.values
np.random.shuffle(data_2)
mod = data_2.shape[0] % n_fold
data_2 = data_2[mod:]

print("---------------------KNN---------------------")
cur = datetime.datetime.now()
accs = cross_validation(data_2, n_fold, k_values, data_type=1, dataframe=df_2)
time.append(datetime.datetime.now() - cur)
for i in range(len(accs)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], accs[i]))

print("---------------------WEIGHTED KNN---------------------")
cur = datetime.datetime.now()
accs = cross_validation(data_2, n_fold, k_values, data_type=1, dataframe=df_2, with_weight=True)
time.append(datetime.datetime.now() - cur)
for i in range(len(accs)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], accs[i]))


# ## Questão 3
# 
# 
# ### Objetivos
# 
# Nesta questão, o objetivo é desenvolver um algoritmo que implemente o classificador k-NN com peso e sem peso utilizando a distância HVDM . Logo após, o desempenho do classificador é avaliado utilizando dois datasets com atributos numéŕicos e categóricos, onde será mostrado o comportamento para diferentes valores de k.
# 
# ### Metodologia
# 
# Para a avaliação do algoritmo proposto pela questão, foram utilizados duas bases de dados obtidas no UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets.html). O primeiro deles foi o "Statlog (German Credit Data)" e o segundo foi o "Fulano de tal".
# As duas bases de dados estavam contidas cada uma em um arquivo .DATA, onde cada linha representa uma instância e as colunas representam os atributos juntamente com as classes.
# Para a leitura deste formato de arquivo foi utilizado a biblioteca Pandas. A biblioteca Matplotlib também foi utilizada para gerar os gráficos utilizados na análise.
# Abaixo segue um resumo da primeira base de dados:

# In[41]:


df_1 = pd.read_csv('datasets/german_credit.data')
pd.set_option('display.max_columns', df_1.shape[1])
df_1.head()


# In[42]:


df_1.describe()


# In[44]:


df_1['C'].value_counts().plot(kind='bar')


# ### Resultados
# 
# Para a avaliação dos classificadores k-NN com e sem peso, cada base de dados foi submetida ao k-fold cross validation, onde o valor escolhido para k foi 10. Cada classificador executou para os números de vizinhos k = {1,2,3,5,7,9,11,13,15}.
# As acurácias obtidas utilizando a primeira base de dados seguem abaixo:

# In[46]:


n_fold = 10
data_1 = df_1.values
np.random.shuffle(data_1)
mod = data_1.shape[0] % n_fold
data_1 = data_1[mod:]

k_values = [1,2,3,5,7,9,11,13,15]
accs = []
weighted_accs = []

print("---------------------KNN---------------------")
cur = datetime.datetime.now()
accs = cross_validation(data_1, n_fold, k_values, data_type=2, dataframe=df_1)
time.append(datetime.datetime.now() - cur)
for i in range(len(accs)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], accs[i]))
    
print("\n---------------------WEIGHTED KNN---------------------")
cur = datetime.datetime.now()
weighted_accs = cross_validation(data_1, n_fold, k_values, data_type=2, dataframe=df_1, with_weight=True)
time.append(datetime.datetime.now() - cur)
for i in range(len(accs)):
    print("The accuracy for k = {:d} is: {:.2f}%".format(k_values[i], weighted_accs[i]))

