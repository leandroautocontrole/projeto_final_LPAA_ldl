#!/usr/bin/env python
# coding: utf-8

# # Projeto FINAL de LPAA - Machine Learning - LEANDRO DANTAS LIMA (059.323.894-00)

# In[1]:


# importando de bibliotecas
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from IPython.display import display, HTML
from scipy import stats
from scipy.stats import f_oneway
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
import geopandas
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import scipy.stats as stats
import warnings


# In[2]:


pip install --upgrade scikit-learn imbalanced-learn


# In[3]:


# desativar mensagens de warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[4]:


# importando o dataset "Average Time Spent By A User On Social Media" para análise
df = pd.read_csv("dummy_data.csv", sep=",", on_bad_lines='skip', low_memory=False)


# In[5]:


# criando uma cópia do dataset para manter o backup do original
df_copy = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[6]:


# mostrando as primeiras linhas para entender os dados
print("Primeiras linhas do dataframe:")
df_copy.head()


# In[7]:


# mostrando as propriedades do df
df_copy.shape


# In[8]:


# mostrando os tipos de dados --> quando não consegue definir, classifica como object
df_copy.info()


# In[9]:


# Estatísticas descritivas
print("\nEstatísticas descritivas:")
df_copy.describe()


# In[10]:


# conferindo e contando se há valores ausentes no df
print("\nValores nulos no dataframe:")
print(df_copy.isnull().sum())


# In[11]:


# conferindo se há dados duplicados
df_copy[df_copy.duplicated()].count().sum()


# In[12]:


# gráfico KDE (Kernel Density Function)
df_copy['time_spent'].plot.kde(subplots = True, figsize = (8,3))


# In[13]:


# Gráfico de Barras dos usuários por países
plt.figure(figsize=(12, 6))
plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
country = df_copy['location'].value_counts()
sns.barplot(x=country.index, y=country.values)
plt.xlabel('Países')
plt.ylabel('Número de Usuários')
plt.title('Localização do usuário')
plt.savefig("country.png") # salvando avistamentos por países
plt.show()


# In[14]:


# gráfico de barras - interesses
plt.figure(figsize=(10,6))
plt.grid(color='lightgrey', linestyle='-', linewidth=0.25)
interests = df_copy['interests'].value_counts()
sns.barplot(x=interests.index, y=interests.values, palette='viridis')
plt.xlabel('Interesses')
plt.xticks(rotation = -30)
plt.ylabel('Quantidade')
plt.title('Interesses dos usuários')
plt.savefig("interests.png") # salvando os interesses
plt.show()


# In[15]:


# histograma - representação gráfica da idade
sns.histplot(df_copy['age'],kde=True)
plt.title('Histograma - Idade dos usuários')
plt.xlabel('Idade')
plt.ylabel('Usuários')
plt.show()


# In[16]:


# histograma - renda dos usuários
sns.histplot(df_copy['income'],kde=True)
plt.title('Histograma - Renda dos usuários')
plt.xlabel('Renda')
plt.ylabel('Usuários')
plt.show()


# In[17]:


# criando uma cópia do dataframe para manter o backup do original
df_copy = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.

# Calculando a matriz de correlação
correlation_matrix = df_copy[['age', 'time_spent', 'income']].corr()

# Visualizando a matriz de correlação como um mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()


# In[18]:


# criando uma cópia do dataframe para manter o backup do original
df_copy2 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[19]:


# calculando ano de nascimento
df_copy2["birth_year"] = 2024 - df_copy2["age"]

# Definindo gerações
bins = [1945, 1964, 1980, 1996, 2012]
labels = ["Baby Boomers", "Generation X", "Millennials", "Generation Z"]

# categorizando de acordo com a geração
df_copy2["generation"] = pd.cut(df_copy2["birth_year"], bins=bins, labels=labels, right=False)


# In[20]:


# descrição do novo dataframe
df_copy2.describe()


# In[21]:


# criando uma cópia do dataframe para manter o backup do original
df_copy3 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[22]:


# selecionando variáveis categóricas
categorical_features = df_copy3.select_dtypes(include=[np.object_, "category"])
for column in categorical_features:
    print(f"Value counts for {column}:")
    print(df_copy3[column].value_counts())
    print("\n")


# In[23]:


# selecionando variáveis booleanas
boolean_features = df_copy3.select_dtypes(include=['bool'])
for column in boolean_features:
    print(f"Value counts for {column}:")
    print(df_copy3[column].value_counts())
    print("\n")


# In[24]:


# Tempo médio gasto em plataformas específicas de mídia social por geração
generation_socialmedia_avgtime = df_copy2.groupby(by=["generation", "platform"]).agg({"time_spent":"mean"}).reset_index()


# In[25]:


# Gráfico de barras do tempo médio gasto nas redes sociais por geração
plt.figure(figsize=(12, 8))
sns.barplot(data=generation_socialmedia_avgtime, x='generation', y='time_spent', hue='platform')
plt.xticks(rotation=0)
plt.title('Tempo médio gasto em plataformas por geração')
plt.xlabel('Geração')
plt.ylabel('Tempo médio gasto (horas)')
plt.legend(title='Platforma', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[26]:


plt.rcParams["figure.autolayout"] = True


# In[27]:


df_copy2.info()


# In[28]:


# definir sementes para reprodutibilidade
np.random.seed(0)


# In[29]:


# criando uma cópia do dataframe para manter o backup do original
df_copy4 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[30]:


# modelo de regressão para prever se uma pessoa está endividada
# Pré-processando os dados - Selecionando recursos e alvo para o modelo de regressão
X = df_copy4.drop(['income', 'indebt', 'isHomeOwner', 'Owns_Car'], axis=1)
y = df_copy4['income']

# Tratamento de variáveis categóricas por OneHotEncoding
categorical_features = ['gender', 'platform', 'interests', 'location', 'demographics', 'profession']
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

# Criando um transformador de coluna para aplicar transformações nas respectivas colunas
preprocessor = ColumnTransformer(transformers=[
    ('cat', one_hot_encoder, categorical_features)
], remainder='passthrough')

# Dividindo o conjunto de dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um pipeline de regressão
regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Treinando o modelo
regression_pipeline.fit(X_train, y_train)

# Previsão no conjunto de testes
y_pred = regression_pipeline.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

mse, rmse, r2
# As métricas de desempenho do modelo de regressão são as seguintes:
# Erro Quadrático Médio (MSE);
# Raiz do erro quadrático médio (RMSE);
# Pontuação R^2.


# In[31]:


# modelo de classificação para prever se uma pessoa está endividada

# Selecionando recursos e destino para o modelo de classificação
X_classification = df_copy4.drop(['income', 'indebt'], axis=1) # Excluding 'income' as it's not a target here
y_classification = df_copy4['indebt']

# Dividindo o conjunto de dados em conjuntos de treinamento e teste para classificação
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Criando um pipeline de classificação
classification_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Treinando o modelo
classification_pipeline.fit(X_train_c, y_train_c)

# Previsão no conjunto de testes
y_pred_c = classification_pipeline.predict(X_test_c)

# avaliando o modelo
accuracy = accuracy_score(y_test_c, y_pred_c)
conf_matrix = confusion_matrix(y_test_c, y_pred_c)

accuracy, conf_matrix
# As métricas de desempenho do modelo de classificação são as seguintes:
# Precisão (acurácia);
# Matriz de confusão:
# Verdadeiros Negativos;
# Falsos Positivos;
# Falsos Negativos;
# Verdadeiros Positivos.


# In[32]:


# criando uma cópia do dataframe para manter o backup do original
df_copy6 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[33]:


# criando função de classificação do tempo de uso
def screen_time(number):
    if number>6:
        return "Extreme"
    elif number>4:
        return "High"
    elif number>2 :
        return "Moderate"
    else :
        return "Normal"


# In[34]:


# criando função de classificação quanto à idade
def life_stage(age):
    if age > 60:
        return "old"
    elif age>=40:
        return "middle_age"
    elif age >= 18:
        return "young"
    else:
        return "teenage"


# In[35]:


df_copy6['life_stage']= df_copy6['age'].apply(life_stage)
df_copy6['screen_time'] = df_copy6['time_spent'].apply(screen_time)


# In[36]:


# gráfico de barras de classificação quanto à idade
sns.countplot(x= df_copy6['life_stage'])


# In[37]:


# gráfico de barras de classificação quanto à idade
sns.countplot(x= df_copy6['screen_time'])


# In[38]:


# gráfico de distribuição bivariada de pares
sns.pairplot(df_copy6[['age','time_spent','income']])


# In[39]:


# criando uma cópia do dataframe para manter o backup do original
df_copy7 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[40]:


# criando estilo para os gráficos com o matplotlib
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(9,5)
matplotlib.rcParams['figure.facecolor']='#00000000'


# In[ ]:





# In[41]:


# Análise exploratória de dados
avg_time_on_sm=df_copy7.groupby(by=['platform']).agg({'time_spent':'mean'}).reset_index()
gender_wise=df_copy7.groupby(by=['gender']).agg({'time_spent':'mean'}).reset_index()
location_wise=df_copy7.groupby(by=['location']).agg({'time_spent':'mean'}).reset_index()

labels=['Australia','United Kingdom','United States']
sizes=[5.218750,4.908815,4.943574]
plt.figure(figsize=(7, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Tempo gasto em termos de localização')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[42]:


demographics_wise=df_copy7.groupby(by=['demographics']).agg({'time_spent':'mean'}).reset_index()
labels=['Rural','Sub_Urban','Urban']
sizes=[5.020588,5.271642,4.787692]
plt.figure(figsize=(7, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Tempo gasto por classificação demografia')
plt.axis('equal')  
plt.show()


# In[ ]:





# In[43]:


profession_wise=df_copy7.groupby(by=['profession']).agg({'time_spent':'mean'}).reset_index()
labels=['Marketer Manager','Software Engineer','Student']
sizes=[5.095775,4.949405,5.038835]
plt.figure(figsize=(7, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Tempo gasto por profissão')
plt.axis('equal')  
plt.show()


# In[ ]:





# In[44]:


professionals_avg_time=df_copy7.groupby(by=['profession','platform']).agg({'time_spent':'mean'}).reset_index()
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(professionals_avg_time.pivot(index='profession', columns='platform', values='time_spent'), annot=True, cmap='YlGnBu')
plt.title('Tempo médio gasto nas redes sociais por profissão e plataforma')
plt.xlabel('Platform')
plt.ylabel('Profession')
plt.show()


# In[45]:


diff_loc=df_copy7.groupby(by=['location','platform']).agg({'time_spent':'mean'}).reset_index()
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(diff_loc.pivot(index='location', columns='platform', values='time_spent'), annot=True, cmap='YlGnBu')
plt.title('Tempo médio gasto nas redes sociais por local e plataforma')
plt.xlabel('Platform')
plt.ylabel('Location')
plt.show()


# In[46]:


labels=['female','male','non-binary']
sizes=[5185.770393,14919.620178,14941.027108]
plt.figure(figsize=(7, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Renda de diferentes gêneros')
plt.axis('equal')  
plt.show()


# In[47]:


correlation_homeowner = df_copy7['time_spent'].corr(df_copy7['isHomeOwner'])
correlation_car_owner = df_copy7['time_spent'].corr(df_copy7['Owns_Car'])

print(f"Correlação entre tempo gasto e propriedade de casa própria: {correlation_homeowner}")
print(f"Correlação entre tempo gasto e propriedade de carro: {correlation_car_owner}")


# In[49]:


# criando uma cópia do dataframe para manter o backup do original
df_copy8 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[50]:


# tempo médio gasto pelos profissionais nas plataformas
profession_platform=pd.DataFrame(df_copy8.groupby('profession')['platform'].value_counts().sort_values(ascending=False).unstack())
profession_time=pd.DataFrame(df_copy8.groupby('profession')['time_spent'].mean())
all_information=pd.merge(profession_platform,profession_time,on='profession')
all_information.style.background_gradient(cmap='ocean')


# In[51]:


# gráfico de "rosca" relacionando as plataformas com as categorias
demographics = ['age', 'gender', 'profession']

for demographic in demographics:
    platform_info = df_copy8.groupby(demographic)['platform'].value_counts().unstack()
    fig = px.pie(platform_info, 
                 names=platform_info.columns, 
                 title=f'Platform Preference by {demographic.capitalize()}',
                 labels={'platform':'Platform'},color_discrete_sequence=px.colors.sequential.RdBu,hole=.3)
    fig.show()
    pyo.iplot(fig)


# In[52]:


# criando uma cópia do dataframe para manter o backup do original
df_copy9 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[53]:


# gráfico de barras Tempo médio gasto em plataformas de mídia social por local
fig=px.bar(df_copy9.groupby(by=['location','platform']).agg({'time_spent':'mean'}).reset_index(),
           x='location',y='time_spent',color='platform',barmode='group',
           title="Tempo médio gasto em plataformas de mídia social por local")
fig.show(render='iframe')


# In[54]:


# histogramas da idade, tempo de uso e renda
plt.figure(figsize=(15,7))
plt.subplot(1,3,1)
sns.histplot(df_copy9.age, kde=True)
plt.subplot(1,3,2)
sns.histplot(df_copy9.time_spent, kde=True)
plt.subplot(1,3,3)
sns.histplot(df_copy9.income, kde=True)
plt.show()


# In[55]:


# criando uma cópia do dataframe para manter o backup do original
df_copy10 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[56]:


# criando uma função para conversão de booleano em binário
def temp(x):
    if x==True:
        return 1
    else:
        return 0

df_copy10.indebt=df_copy10.indebt.apply(temp)
df_copy10.isHomeOwner=df_copy10.isHomeOwner.apply(temp)
df_copy10.Owns_Car=df_copy10.Owns_Car.apply(temp)


# In[57]:


# correlação entre as variáveis
df_copy10[['age', 'time_spent', 'income', 'indebt', 'isHomeOwner', 'Owns_Car']].corr()


# In[58]:


# mapa de calor da correlação entre as variáveis
sns.heatmap(df_copy10[['age', 'time_spent', 'income', 'indebt', 'isHomeOwner', 'Owns_Car']].corr(), cmap='Blues')
plt.show()


# In[60]:


# gráficos de barras relacionando a utilização das plataformas nos países por gênero
sns.catplot(data=df_copy10, col='platform', hue='gender', x='location', kind='count')
plt.show()


# In[62]:


# criando classificações para idade e renda
np.linspace(18, 64, 4)
def age_group(x):
    if x<34:
        return 'Young'
    elif x>48:
        return 'Old'
    else:
        return 'Middle'

df_copy10['age_group']=df_copy10.age.apply(age_group)

np.linspace(10012, 19980, 4)
def income_group(x):
    if x<13335:
        return 'lower'
    elif x>16657:
        return 'higher'
    else:
        return 'middle'

df_copy10['income_group']=df_copy10.income.apply(income_group)


# In[63]:


# mapa de calor dos interesses por idade e renda
sns.heatmap(pd.crosstab(df_copy10.interests, [df_copy10.age_group, df_copy10.income_group]), annot=True, cmap='Blues')
plt.show()


# In[64]:


# normalizando os dados de interesse, profissão e plataforma
pd.crosstab(df_copy10.interests, [df_copy10.profession, df_copy10.platform], margins=True, normalize=True)


# In[65]:


# normalizando os dados de interesse, profissão e plataforma
pd.crosstab(df_copy10.interests, [df_copy10.profession, df_copy10.platform], margins=True, normalize='index')


# In[66]:


# normalizando os dados de interesse, profissão e plataforma
pd.crosstab(df_copy10.interests, [df_copy10.profession, df_copy10.platform], margins=True, normalize='columns')


# In[68]:


# tabela cruzada relacionando plataforma, localização e classificação demográfica
pd.crosstab(df_copy10.platform, [df_copy10.location, df_copy10.demographics])


# In[69]:


# mapa de calor com dados da tabela cruzada
sns.heatmap(pd.crosstab(df_copy10.platform, [df_copy10.location, df_copy10.demographics]), annot=True, cmap='Blues')
plt.show()


# In[70]:


# tabela cruzada normalizada
pd.crosstab(df.platform, [df_copy10.location, df_copy10.demographics], margins=True, normalize=True)


# In[71]:


# tabela cruzada normalizada
pd.crosstab(df_copy10.platform, [df_copy10.location, df_copy10.demographics], margins=True, normalize='index')


# In[72]:


# tabela cruzada normalizada
pd.crosstab(df_copy10.platform, [df_copy10.location, df_copy10.demographics], margins=True, normalize='columns')


# In[74]:


# análise financeira
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
sns.histplot(data=df_copy10, x='income', kde=True)
plt.subplot(1,2,2)
sns.countplot(data=df_copy10, x='income_group')
plt.show()


# In[76]:


# tabela cruzada renda, endividamento, posses de casas e de carros
pd.crosstab(df_copy10.income_group, [df_copy10.indebt, df_copy10.isHomeOwner, df_copy10.Owns_Car], margins=True)


# In[77]:


# mapa de calor d atabela cruzada
sns.heatmap(pd.crosstab(df_copy10.income_group, [df_copy10.indebt, df_copy10.isHomeOwner, df_copy10.Owns_Car]), cmap='Blues', annot=True)
plt.show()


# In[79]:


# gráficos de barras relacionando interesses e plataformas com o nível de renda
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
sns.countplot(data=df_copy10, x='income_group', hue='interests')
plt.subplot(1,2,2)
sns.countplot(data=df_copy10, x='income_group', hue='platform')
plt.show()


# In[81]:


# tabela cruzada renda, plataforma e interesses normalizada
pd.crosstab(df_copy10.income_group, [df_copy10.platform, df_copy10.interests], margins=True, normalize='all')


# In[82]:


# tabela cruzada renda, plataforma e interesses normalizada
pd.crosstab(df_copy10.income_group, [df_copy10.platform, df_copy10.interests], margins=True, normalize=0)


# In[83]:


# mapa de calor da tabela cruzada
sns.heatmap(pd.crosstab(df_copy10.income_group, [df_copy10.platform, df_copy10.interests]), cmap='Blues')
plt.show()


# In[85]:


# criando uma cópia do dataframe para manter o backup do original
df_copy11 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[86]:


# colunas categóricas
categorical_columns=["gender","platform","interests","location","demographics","profession"]


# In[87]:


# mapas de calor com relação ao gênero
ig, axs = plt.subplots(3, 2, figsize=(10,15))

for i, col in enumerate(categorical_columns):
    cross_tab = pd.crosstab(df_copy11[col], df_copy11["gender"])    
    sns.heatmap(cross_tab, ax=axs[i // 2, i % 2], cmap='coolwarm', annot=True, fmt='d')
    
    axs[i // 2, i % 2].set_title(f'Mapa de calor {col} vs Gênero')
    axs[i // 2, i % 2].set_xlabel('Gênero')
    axs[i // 2, i % 2].set_ylabel(col.capitalize())

plt.tight_layout()
plt.show()


# In[88]:


# mapas de calor com relação à plataforma
ig, axs = plt.subplots(3, 2, figsize=(10, 15))

for i, col in enumerate(categorical_columns):
    cross_tab = pd.crosstab(df_copy11[col], df_copy11["platform"])
    sns.heatmap(cross_tab, ax=axs[i // 2, i % 2], cmap='coolwarm', annot=True, fmt='d')
    
    axs[i // 2, i % 2].set_title(f'Mapa de Calor {col} vs Plataforma')
    axs[i // 2, i % 2].set_xlabel('Plataforma')
    axs[i // 2, i % 2].set_ylabel(col.capitalize())

plt.tight_layout()
plt.show()


# In[89]:


# distribuição de tempomédio gasto por categorias
categorical_features = df_copy11.select_dtypes(include=['object', 'bool'])

for column in categorical_features:
    plt.figure(figsize=(8, 6))
    avg_time_spent = df_copy11.groupby(column)['time_spent'].mean()
    df_copy11[column].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.text(-1.5, 1, 'Tempo médio gasto:', fontsize=12, fontweight='bold')
    for i, (index, value) in enumerate(avg_time_spent.items()):
        plt.text(-1.5, 0.9-i*0.2, f"{index}: {value:.2f} min", fontsize=10)
    plt.title('Distribuição por {}'.format(column))
    plt.ylabel('')
    plt.show()


# In[90]:


# criando uma cópia do dataframe para manter o backup do original
df_copy12 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[91]:


# modelo de classificação por RandonForest
y = df_copy12["age"]

features = ['isHomeOwner', 'Owns_Car',"indebt",'income']
X = pd.get_dummies(df_copy12[features])

# divisão de treinamento/validação
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# treinamento
test_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
test_model.fit(train_X, train_y)
predictions = test_model.predict(train_X)

# geração de previsão inicial
print("First in-sample predictions:", test_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# validação
val_predictions = test_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

# função get_mae
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# árvore de candidatos
candidate_max_leaf_nodes = range(2,100,2)

for max_leaf_nodes in candidate_max_leaf_nodes :
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# Armazena o melhor valor (5, 25, 50, 100, 250 ou 500)
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

# Cálculo do Modelo Final
final_model = RandomForestClassifier(max_leaf_nodes = best_tree_size, random_state=1)
final_model.fit(X, y)
print("First in-sample predictions:", final_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())
print(final_model.predict(X))
print(y)


# In[92]:


# criando uma cópia do dataframe para manter o backup do original
df_copy13 = df.copy(deep=True)  # deep=True (padrão) o novo objeto será criado com uma cópia dos dados e índices do objeto original, sem alterações no original.


# In[93]:


# regressão linear

count = df_copy13['age'].value_counts()

# selecionando colunas categóricas
categorical_cols = ['gender', 'platform', 'interests', 'location', 'demographics', 'profession']

df_encoded = pd.get_dummies(df_copy13, columns=categorical_cols)

# Exibe as primeiras linhas do DataFrame codificado
df_encoded.head()

# separando as variáveis
X = df_encoded.drop(columns=['time_spent'])
y = df_encoded['time_spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# imprimindo os conjuntos de treinamento e teste
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Inicializndo o modelo de regressão linear
model = LinearRegression()

# Ajustando o modelo aos dados de treinamento
model.fit(X_train, y_train)

# Prevendo os dados de teste
y_pred = model.predict(X_test)

# avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

model1 = RandomForestRegressor()
model.fit(X_train, y_train)

# Fazendo previsões e avaliando o modelo
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[94]:


# Gradient Boosting Regression
r2 =r2_score(y_test,y_pred) 

# Inicializando o modelo Gradient Boosting Regression
gb_model = GradientBoostingRegressor(random_state=42)

# Ajustando o modelo aos dados de treinamento
gb_model.fit(X_train, y_train)

# Prevendo os dados de teste
gb_y_pred = gb_model.predict(X_test)

# avaliando o modelo
gb_mse = mean_squared_error(y_test, gb_y_pred)
gb_r2 = r2_score(y_test, gb_y_pred)

print("Gradient Boosting Regression:")
print("Mean Squared Error:", gb_mse)
print("R-squared Score:", gb_r2)


# In[95]:


# SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializando o modelo de regressão do vetor de suporte
svr_model = SVR()

# Ajustando o modelo aos dados de treinamento
svr_model.fit(X_train_scaled, y_train)

# Prevendo os dados de teste
svr_y_pred = svr_model.predict(X_test_scaled)

# avaliando o modelo
svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)

print("Support Vector Regression:")
print("Mean Squared Error:", svr_mse)
print("R-squared Score:", svr_r2)


# In[96]:


# Crie recursos de interação
df_encoded['age_income_interaction'] = df_encoded['age'] * df_encoded['income']
df_encoded['age_time_spent_interaction'] = df_encoded['age'] * df_encoded['time_spent']
df_encoded['income_time_spent_interaction'] = df_encoded['income'] * df_encoded['time_spent']

df_encoded.head()


# In[99]:


# variável polinomial
poly_features = PolynomialFeatures(degree=2, include_bias=False)
poly_data = poly_features.fit_transform(X)
poly_feature_names = poly_features.get_feature_names_out(input_features=X.columns)

# criando dataframe com recursos polinomiais
df_poly = pd.DataFrame(poly_data, columns=poly_feature_names)
df_poly = pd.concat([df_encoded, df_poly], axis=1)


# In[100]:


# Split the dataset into training and testing sets (assuming df_poly contains the polynomial features)
X_poly_train, X_poly_test, y_train, y_test = train_test_split(df_poly, y, test_size=0.2, random_state=42)

# Initialize and train the models
linear_model = LinearRegression()
linear_model.fit(X_poly_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_poly_train, y_train)

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_poly_train, y_train)

svr_model = SVR()
svr_model.fit(X_poly_train, y_train)

# Evaluate the models
models = {
    "Linear Regression": linear_model,
    "Random Forest Regression": rf_model,
    "Gradient Boosting Regression": gb_model,
    "Support Vector Regression": svr_model
}

for name, model in models.items():
    y_pred = model.predict(X_poly_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    print()

