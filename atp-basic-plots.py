#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install plotly')
get_ipython().system('pip install numpy')


# In[2]:


import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


# In[3]:


data = pd.read_csv("./ATP.csv", encoding= "latin_1")
data.shape


# In[4]:


# Tecnica de visualizacion de datos: Ver cuantos partidos han sido jugados en "x" tipos canchas
tm = data['surface'].value_counts(normalize = True)*100
print(tm)

plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=500,
    height=500,
    title = "Numero de superficies en canchas"
)
fig = go.Figure(data=plt_data, layout=layout)
# Lo de abajo esta funcionando, pero no funciona en colab google por alguna razon
fig.show()
#iplot(fig)


# In[5]:


#Graficar el numero de encuentros en donde ha habido vs de RR RL LR LL
data.loc[(data.winner_hand == 'R') & (data.loser_hand == 'R') , 'versus'] = 'R vs R' 
data.loc[(data.winner_hand == 'R') & (data.loser_hand == 'L') , 'versus'] = 'R vs L'
data.loc[(data.winner_hand == 'L') & (data.loser_hand == 'R') , 'versus'] = 'R vs L'
data.loc[(data.winner_hand == 'L') & (data.loser_hand == 'L') , 'versus'] = 'L vs L'

tm = data['versus'].value_counts(normalize = True)*100
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=500,
    height=500,
    title = "Encuentros entre Diestros y zurdos"
)
fig = go.Figure(data=plt_data, layout=layout)
iplot(fig)


# In[6]:


# Right or Left - which is better?
# Solo obtenemos las filas que tengan definido L o R en el jugador ganador o perdedor
data = data[(data.loser_hand == 'R') | (data.loser_hand == 'L')]
data = data[(data.winner_hand == 'R') | (data.winner_hand == 'L')] 

# Las veces que ha ganado R sobre L
R = len(data[(data.winner_hand == 'R') & (data.loser_hand == 'L') ])
# Las veces que ha ganado L sobre R
L = len(data[(data.winner_hand == 'L') & (data.loser_hand == 'R')])

print(f'Un jugador diestro tiene un { round((R/(R+L))*100,2) }% de ganar')
print(f'Un jugador zurdo tiene un { round((L/(R+L))*100,2) }% de ganar')

# De aqui sacamos los porcentajes de los ganadores especialistas con la mano derecha y la mano izquierda
data.loc[(data.winner_hand == 'R') & (data.loser_hand == 'L'), 'winner_match'] = 'Diestro'
data.loc[(data.winner_hand == 'L') & (data.loser_hand == 'R'), 'winner_match'] = 'Zurdo'

tm = data['winner_match'].value_counts(normalize = True)*100
plt_data = [go.Bar(
    x = tm.index,
    y = tm
    )]
layout = go.Layout(
    autosize=False,
    width=500,
    height=500,
    title = "Ganadores diestros vs zurdos"
)
fig = go.Figure(data=plt_data, layout=layout)
fig.show()


# In[7]:


def normalizacionColumna(data,i):
    columns = data.columns.values
    data[columns[i]] = (data[columns[i]] - data[columns[i]].min()) / (data[columns[i]].max() - data[columns[i]].min())


# In[8]:


def normalizarDataset(data,indices):
    df = data.copy()
    for i in indices:
        normalizacionColumna(df,i)
    return df


# In[9]:


# draw_size, winner_ht, loser_ht, winner_age, loser_age, winner_rank, loser_rank, winner_rank_points, 
# minutes
indices = [4]
data = normalizarDataset(data,indices)


# In[10]:


# surface lo puedo categorizar
# tourney_level lo puedo categorizar

data.columns


# In[11]:


data['winner_match'].value_counts()


# In[ ]:


# IDEAS 
## Graficar contries winner_ioc


# In[16]:


def weightedAverage(data, w, l):
    df = data.copy()
    columns = df.columns.values
    wa = np.zeros(df.shape[0])
    for i in range(len(l)):
        wa += (w[i] * df[columns[l[i]]]) / sum(w)
    df['wa'] = wa
    df = df.sort_values(by=['wa'], ascending=False)
    return df


# In[124]:


data_winner = data[['matchid','winner_name','winner_ht','winner_age','winner_hand','winner_rank']]
data_loser = data[['matchid','loser_name','loser_ht','loser_age','loser_hand','loser_rank']]


# In[125]:


data_winner.isna().sum()


# In[126]:


data_loser.isna().sum()


# In[127]:


data_winner.dropna(subset = ['winner_age','winner_ht','winner_hand','winner_rank'],inplace = True)
data_loser.dropna(subset = ['loser_age','loser_ht','loser_hand','loser_rank'],inplace = True)


# In[128]:


data_winner.isna().sum()


# In[129]:


data_loser.isna().sum()


# In[130]:


data_loser


# In[131]:


data_winner


# In[132]:


indices = [2,3,5]
data_winner = normalizarDataset(data_winner,indices)
data_loser = normalizarDataset(data_loser,indices)


# In[133]:


data_winner['winner_hand'] = data_winner['winner_hand'].apply(lambda x: 1 if x == 'R' else 0)
data_loser['loser_hand'] = data_loser['loser_hand'].apply(lambda x: 1 if x == 'R' else 0)


# In[134]:


data_loser


# In[135]:


data_winner


# In[171]:


indices_winner = [2,3,4,5]

pesos = [1,1,3,3]

dfWA_winner = weightedAverage(data_winner, pesos, indices_winner)
dfWA_winner.head(5)


# In[170]:


indices_loser = [2,3,4,5]
pesos = [1,1,3,3]
dfWA_loser = weightedAverage(data_loser, pesos, indices_loser)
dfWA_loser.head(5)


# In[196]:


def maximin(data, l):
    df = data.copy()
    columns = df.columns.values
    print(columns)
    t = df.shape[0]
    print(t)
    mn = np.zeros(df.shape[0])
    print('mn',mn)
    for i in range(t):
        print(i,df[columns[2]])
        print(mn[i],df[columns[l[0]]][i])
        mn[i] = df[columns[l[0]]][i]
        for j in range(1,len(l)):
            if mn[i] > df[columns[l[j]]][i]:                
                mn[i] = df[columns[l[j]]][i]
    df['minVal'] = mn
    df = df.sort_values(by=['minVal'], ascending=False)
    return df


# In[197]:



dfMM_winner = maximin(data_winner, indices_winner)

# data_winner['winner_hand'] = data_winner['winner_hand'].apply(lambda x: float(x))
# data_winner

dfMM_winner.head(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




