import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import mlxtend
import os
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
#import time
import lightfm
from datetime import datetime,  timedelta, date
from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#007efd', '#FFC000', '#303030'])

from funk_svd import SVD
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from surprise import KNNWithMeans
from surprise import Dataset, NormalPredictor, Reader
from surprise.model_selection import cross_validate

from lightfm.data import Dataset as DT
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score as auc_score_lfm

from rexmex.metrics.coverage import item_coverage
from rexmex.metrics.ranking import mean_reciprocal_rank, personalization

from pathlib import Path
from streamlit.source_util import (
    page_icon_and_name, 
    calc_md5, 
    get_pages,
    _on_pages_changed
)
from streamlit.components.v1 import html

# create navigation
def nav_page(page_name, timeout_secs=3):
    #credits: https://stackoverflow.com/questions/73755240/streamlit-pages-on-button-press-not-on-sidebar
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)


def delete_page(main_script_path_str, page_name):

    current_pages = get_pages(main_script_path_str)

    for key, value in current_pages.items():
        if value['page_name'] == page_name:
            del current_pages[key]
            break
        else:
            pass
    _on_pages_changed.send()

def add_page(main_script_path_str, page_name):
    
    pages = get_pages(main_script_path_str)
    main_script_path = Path(main_script_path_str)
    pages_dir = main_script_path.parent / "pages"
    script_path = [f for f in pages_dir.glob("*.py") if f.name.find(page_name) != -1][0]
    script_path_str = str(script_path.resolve())
    pi, pn = page_icon_and_name(script_path)
    psh = calc_md5(script_path_str)
    pages[psh] = {
        "page_script_hash": psh,
        "page_name": pn,
        "icon": pi,
        "script_path": script_path_str,
    }
    _on_pages_changed.send()

def extract_hour(release_date):
  if type(release_date) == str:
    at=datetime.strptime(release_date, "%Y-%m-%d %H:%M:%S.%f")
    return at.strftime("%H:%M")
  else:
    df_loja_rec.dth_agendamento=df_loja_rec.dth_agendamento.astype('str')
    at=datetime.strptime(release_date, "%Y-%m-%d %H:%M:%S.%f")
    return at.strftime("%H:%M")

def time_filter(df, hr=datetime.now(),nh=0):
    df['dth_hora'] = pd.to_datetime(df['dth_hora'])
    ay=df.dth_hora.dt.year.max()
    am=df.dth_hora.dt.month.max()
    ad=df.dth_hora.dt.day.max()
    up_l=datetime(ay,am,ad,hr.hour,hr.minute)+timedelta(hours=nh)
    low_l=datetime(ay,am,ad,hr.hour,hr.minute)-timedelta(hours=nh)
    #dfr=df[(df.dth_hora>=(datetime(ay,am,ad,hr.hour,hr.minute)-timedelta(hours=2)))&(df.dth_hora<=(datetime(ay,am,ad,hr.hour,hr.minute)+timedelta(hours=2)))].copy()
    dfr=df[(df['dth_hora']>=low_l)&(df['dth_hora']<=up_l)].copy()
    dfr.dth_hora=dfr.dth_hora.astype('str')
    dfr.dth_hora.apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%H:%M"))
    return dfr


def f_escolha(df):

    delete_page("teste_strealit_main.py", "shop")
    delete_page("teste_strealit_main.py", "cart")

    st.title("Bem vindo!")

    st.write("Este é um ambiente de testes para sistemas de recomendação")
        #Seleção da loja 
    
    store = st.selectbox(
        'Selecione a Loja:',
        df['loja_compra'].unique())

    ##Seleção do usuário:
    user= st.selectbox(
        'Selecione o usuário:',
        df[df['loja_compra']==store]['cliente_nome'].unique())

    if 'df' not in st.session_state:
        st.session_state.df=df
    if 'store' not in st.session_state:
        st.session_state.store=store
    if 'user' not in st.session_state:
        st.session_state.user=user

    if st.button('Login'):
        st.session_state.df=df
        st.session_state.store=store
        st.session_state.user=user
        add_page('teste_strealit_main.py', 'shop')
        nav_page('shop')

    #movido para o script da segunda pagina '2_shop.py'
    ##Seleção dos campos referente ao produto:
    #st.write('Selecione o produto para o carrinho:')
    df_loja=df[df['loja_compra']==store]
    df_loja_recnp=df_loja.copy()
    df_loja_recnp['produto_f']=df_loja_recnp['produto']+" "+df_loja_recnp['prodcomplemento']

    ##Seleção da categoria do produto
    #cat = st.selectbox(
    #    'Selecione a categoria:',
    #    df_loja.categoria.unique())
    #df_cat=df_loja[df_loja['categoria']==cat]

    ##Seleção do tipo do produto
    #tipo = st.selectbox(
    #    'Selecione o tipo:',
    #    df_cat.tipo_categoria.unique())
    #df_tipo=df_cat[df_cat['tipo_categoria']==tipo]
    #Seleção do produto
    #product=st.selectbox(
    #    'Selecione o produto:',
    #    df_tipo.produto.unique())
    #df_prod=df_tipo[df_tipo['produto']==product]
    #Seleção do complemento

    #if df_prod.prodcomplemento.isin([""]).count()>=1 and len(df_prod.prodcomplemento.unique())==1:
    #        p_dis=True
    #        p_vis="collapsed"
    #else:
    #    p_dis=False
    #    p_vis="visible"

    #complement=st.selectbox(
    #    'Selecione o complemento:',
    #    df_prod.prodcomplemento.unique(),
    #    disabled=p_dis,
    #    label_visibility=p_vis)

    #df_compl=df_prod[df_prod['prodcomplemento']==complement]

    #prodf=product+" "+str(complement)


    #if st.button('Add carrinho'):
    #        st.write(prodf,"adicionado ao carrinho.")
    #        st.session_state.l_prod.append(prodf)
    #if st.button('Carrinho'):
    #    add_page('teste_strealit_main.py', 'cart')
    #    nav_page('cart')

    ###########################################################
    #movido para o script da segunda pagina '3_cart.py'
    #if len(st.session_state.l_prod)==0:
    #    state=True
    #else:
    #    state=False
    #if st.button('Del item',disabled=state):
    #    if len(st.session_state.l_prod)==1:
    #        st.write (f"{st.session_state.l_prod[-1]}, removido do carrinho.")
    #        st.session_state.l_prod=[]
    #        placeholder.empty()
    #    else:    
    #        st.write (f"{st.session_state.l_prod[-1]}, removido do carrinho.")
    #        st.session_state.l_prod.pop()
    #        placeholder.empty()  
    #        placeholder.text("Carrinho:")
    #        with placeholder.container():
    #            st.write('Carrinho:')
    #            for i in st.session_state.l_prod[0:-2]:
    #                st.write(i)
    #
    #if st.button('Del carrinho',disabled=state):
    #    st.session_state.l_prod=[]
    #    st.write (f"Carrinho limpo.") 
    #    placeholder.empty()


    return df_loja_recnp

def main():
    if 'l_prod' not in st.session_state:
        st.session_state.l_prod = []
    if 'df_lrecnp' not in st.session_state:
        st.session_state.df_lrecnp=pd.DataFrame()
    #lista de produtos no carrinho
    #df = pd.read_csv(r"C:\Users\vitor\Documents\Python\streamlit\Scripts\output.csv", encoding = 'utf-8')
    df_server= pd.read_csv(r"https://github.com/vpbatista85/epd/blob/main/output.csv?raw=true", encoding = 'utf-8')
    #creating parquet file to try be faster on execution:
    df_server.to_parquet('df.parquet.gzip',compression='gzip')
    df=pd.read_parquet('df.parquet.gzip')
    #df=df_server.copy()
    df.drop_duplicates(inplace=True)
    df.fillna("",inplace=True)
    st.session_state.df_lrecnp=f_escolha(df)
    #f_carrinho()
    #r_np(df_loja_recnp,st.session_state.l_prod)
    #r_p(df_loja_recnp,st.session_state.l_prod)
    st.session_state.df_lrecnp['dth_hora']=st.session_state.df_lrecnp['dth_agendamento'].apply(extract_hour)
    


def f_carrinho():
        import streamlit as st
        placeholder = st.empty()
        placeholder.text("Carrinho:")

        with placeholder.container():
            st.write('Carrinho:')
            if len(st.session_state.l_prod)<1:
                st.write('O carrinho está vazio.')
            else:
                for i in st.session_state.l_prod:
                    st.write(i)

def rnp_apr(dfs:pd.DataFrame,l_prod,n:int):
    #Recomendação não personalizada utilizando o algoritimo apriori.
    ##agrupando os pedidos
    df_l=dfs[['cod_pedido','produto_f']].groupby('cod_pedido').agg({'produto_f': lambda x : ','.join(set(x))})
    df_l.rename(columns={'produto_f':'itens'},inplace=True)
    df_l.reset_index(inplace=True)
    df_l['itens']=df_l.itens.str.split(pat=',')
    df_l.head()
    ##aplicando as funções do mlxtend:
    encoder=TransactionEncoder()
    te_array=encoder.fit(list(df_l.itens)).transform(list(df_l.itens))
    dft=pd.DataFrame(data=te_array,columns=encoder.columns_)
    frequent_items=apriori(dft,min_support=0.01,use_colnames=True)
    frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))
    try:
        rules=association_rules(frequent_items, metric='lift',min_threshold=1.0)
        rules.sort_values(by='lift',ascending=False)
        rules.antecedents=rules.antecedents.astype('string')
        rules.consequents=rules.consequents.astype('string')
        rules.antecedents=rules.antecedents.str.strip('frozenset({})')
        rules.consequents=rules.consequents.str.strip('frozenset({})')
        #recomendação
        recommendations=pd.DataFrame(columns=rules.columns)
        for i in l_prod:
            recommendations=pd.concat([recommendations,rules[rules.antecedents.str.contains(i, regex=False)]],ignore_index=True)
        for i in l_prod:
            recommendations=recommendations[recommendations.consequents.str.contains(i, regex=False)==False]
        recommendations.consequents.drop_duplicates(inplace=True)
    except ValueError:
        recommendations=pd.DataFrame()

   
    return recommendations.head(n)

def rnp_top_n(ratings:pd.DataFrame, l_prod:list, n:int) -> pd.DataFrame:
    #Recomendação não personalizada por n produtos mais consumidos.
    recommendations = (
        ratings
        .groupby('produto_f')
        .count()['cliente_nome']
        .reset_index()
        .rename({'cliente_nome': 'score'}, axis=1)
        .sort_values(by='score', ascending=False)
    )
    if l_prod != None:
        for i in l_prod:
            recommendations=recommendations[recommendations.produto_f.str.contains(i, regex=False)==False]
    return recommendations.head(n)  

def rnp_cb(df:pd.DataFrame,df_f:pd.DataFrame,l_prod:list,n:int)-> pd.DataFrame:
    #preparando o dataframe para aplicação do algoritimo:
    try:
        dfl=df.reset_index()
        dfl['produto_full']=dfl['categoria']+" "+dfl['tipo_categoria']+" "+dfl['produto']+" "+dfl['prodcomplemento']
        dfl['produto_f']=dfl['produto']+" "+dfl['prodcomplemento']
        BASE_FEATURES=['index','produto_f','produto_full']

        #definindo os vetores maximamente esparços:
        df_gd=pd.get_dummies(dfl[['categoria','tipo_categoria','produto','prodcomplemento']])

        #unindo ao df da loja (dfl):
        df_l=dfl[BASE_FEATURES].merge(df_gd,left_index=True,right_index=True)

        #agrupando por pelo nome completo  para encontrar a matriz maximamente esparça dos items:
        df_ll=df_l.groupby('produto_full').max()
        df_ll.set_index('index',inplace=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        df_ll.index.name = 'id'
        df_ll.index=df_ll.index.astype(str)
        df_ll.columns=df_ll.columns.astype(str)
        df_train=df_ll.iloc[:,1:]
        pipeline = Pipeline([('scaler', MinMaxScaler())])
        pipeline.fit(df_train)

        #gerando a representação vetorial:
        df_vectors = pd.DataFrame(pipeline.transform(df_train))
        df_vectors.columns = df_train.columns
        df_vectors.index = df_train.index
        df_vectors.index.name = 'id'

        #Calculando a matriz de similaridade item-item:
        similarity_matrix = pd.DataFrame(cosine_similarity(df_vectors))
        similarity_matrix.index = df_vectors.index.astype(str)
        similarity_matrix.index.name = 'id'
        similarity_matrix.columns = df_vectors.index.astype(str)
        recommendations=pd.DataFrame(columns=similarity_matrix.columns)
        for i in l_prod:
            #a=df_ll[df_ll['produto_f']==i].index[0]
            item_id=df_ll[df_ll['produto_f']==i].index[0]
            #Gerando recomendações
            target_item_similarities = similarity_matrix.loc[item_id]
            id_similar_items = (
                target_item_similarities
                .sort_values(ascending=False)
                .reset_index()
                .rename({'index': 'id', item_id: 'score'}, axis=1)
            )
            r=id_similar_items.merge(df_ll[['produto_f']],left_on='id',right_on='id',how='inner').sort_values(by='score', ascending=False)
            if len(l_prod)>1:
                recommendations=pd.concat([recommendations,r[1:2]])
            else:
                recommendations=pd.concat([recommendations,r[1:5]])
    except (IndexError,ValueError) as e:
        dfl=df_f.reset_index()
        dfl['produto_full']=dfl['categoria']+" "+dfl['tipo_categoria']+" "+dfl['produto']+" "+dfl['prodcomplemento']
        dfl['produto_f']=dfl['produto']+" "+dfl['prodcomplemento']
        BASE_FEATURES=['index','produto_f','produto_full']

        #definindo os vetores maximamente esparços:
        df_gd=pd.get_dummies(dfl[['categoria','tipo_categoria','produto','prodcomplemento']])

        #unindo ao df da loja (dfl):
        df_l=dfl[BASE_FEATURES].merge(df_gd,left_index=True,right_index=True)

        #agrupando por pelo nome completo  para encontrar a matriz maximamente esparça dos items:
        df_ll=df_l.groupby('produto_full').max()
        df_ll.set_index('index',inplace=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        df_ll.index.name = 'id'
        df_ll.index=df_ll.index.astype(str)
        df_ll.columns=df_ll.columns.astype(str)
        df_train=df_ll.iloc[:,1:]
        pipeline = Pipeline([('scaler', MinMaxScaler())])
        pipeline.fit(df_train)

        #gerando a representação vetorial:
        df_vectors = pd.DataFrame(pipeline.transform(df_train))
        df_vectors.columns = df_train.columns
        df_vectors.index = df_train.index
        df_vectors.index.name = 'id'

        #Calculando a matriz de similaridade item-item:
        similarity_matrix = pd.DataFrame(cosine_similarity(df_vectors))
        similarity_matrix.index = df_vectors.index.astype(str)
        similarity_matrix.index.name = 'id'
        similarity_matrix.columns = df_vectors.index.astype(str)
        recommendations=pd.DataFrame(columns=similarity_matrix.columns)
        for i in l_prod:
            #a=df_ll[df_ll['produto_f']==i].index[0]
            item_id=df_ll[df_ll['produto_f']==i].index[0]
            #Gerando recomendações
            target_item_similarities = similarity_matrix.loc[item_id]
            id_similar_items = (
                target_item_similarities
                .sort_values(ascending=False)
                .reset_index()
                .rename({'index': 'id', item_id: 'score'}, axis=1)
            )
            r=id_similar_items.merge(df_ll[['produto_f']],left_on='id',right_on='id',how='inner').sort_values(by='score', ascending=False)
            if len(l_prod)>1:
                recommendations=pd.concat([recommendations,r[1:2]])
            else:
                recommendations=pd.concat([recommendations,r[1:5]])

    return recommendations.head(n)         

def rp_cv(df:pd.DataFrame,df_f:pd.DataFrame, l_prod:list, n:int)-> pd.DataFrame:
    try:
        #preparando o dataframe para aplicação do algoritimo:
        dflg=df.reset_index()
        dflg['produto_full']=dflg['categoria']+" "+dflg['tipo_categoria']+" "+dflg['produto']+" "+dflg['prodcomplemento']
        recommendations=pd.DataFrame(columns=['item_id', 'score'])
        
        #criando o grafo:
        n_users = dflg['cliente_nome'].unique()
        n_items = dflg['produto_f'].unique()
        G = nx.Graph()
        G.add_nodes_from(n_items, node_type='item')
        G.add_nodes_from(n_users, node_type='user')
        G.add_edges_from(dflg[['cliente_nome','produto_f']].values)
        recommendations=pd.DataFrame(columns=['item_id', 'score'])
        for i in l_prod:
            item_id=i
            #Encontrando os itens vizinhos consumidos:
            neighbors = G.neighbors(item_id)
            neighbor_consumed_items = []
            for user_id in neighbors:
                user_consumed_items = G.neighbors(user_id)
                neighbor_consumed_items += list(user_consumed_items)

            #Contabilizando os items mais consumidos para criar o score da recomendação:
            consumed_items_count = Counter(neighbor_consumed_items)

            # Validando tipo do nó
            node_type = nx.get_node_attributes(G, 'node_type')[item_id]
            if node_type != 'item':
                raise ValueError('Node is not of item type.')

            # Contabilizando itens consumidos pelos vizinhos
            consumed_items_count = Counter(neighbor_consumed_items)

            # Criando dataframe
            df_neighbors= pd.DataFrame(zip(consumed_items_count.keys(), consumed_items_count.values()))
            df_neighbors.columns = ['item_id', 'score']
            df_neighbors = df_neighbors.sort_values(by='score', ascending=False)
            
            if len(l_prod)>1:
                recommendations=pd.concat([recommendations,df_neighbors[1:2]])
            else:
                recommendations=pd.concat([recommendations,df_neighbors[1:6]])
    except nx.exception.NetworkXError:
                #preparando o dataframe para aplicação do algoritimo:
        dflg=df_f.reset_index()
        dflg['produto_full']=dflg['categoria']+" "+dflg['tipo_categoria']+" "+dflg['produto']+" "+dflg['prodcomplemento']
        recommendations=pd.DataFrame(columns=['item_id', 'score'])
        
        #criando o grafo:
        n_users = dflg['cliente_nome'].unique()
        n_items = dflg['produto_f'].unique()
        G = nx.Graph()
        G.add_nodes_from(n_items, node_type='item')
        G.add_nodes_from(n_users, node_type='user')
        G.add_edges_from(dflg[['cliente_nome','produto_f']].values)
        recommendations=pd.DataFrame(columns=['item_id', 'score'])
        for i in l_prod:
            item_id=i
            #Encontrando os itens vizinhos consumidos:
            neighbors = G.neighbors(item_id)
            neighbor_consumed_items = []
            for user_id in neighbors:
                user_consumed_items = G.neighbors(user_id)
                neighbor_consumed_items += list(user_consumed_items)

            #Contabilizando os items mais consumidos para criar o score da recomendação:
            consumed_items_count = Counter(neighbor_consumed_items)

            # Validando tipo do nó
            node_type = nx.get_node_attributes(G, 'node_type')[item_id]
            if node_type != 'item':
                raise ValueError('Node is not of item type.')

            # Contabilizando itens consumidos pelos vizinhos
            consumed_items_count = Counter(neighbor_consumed_items)

            # Criando dataframe
            df_neighbors= pd.DataFrame(zip(consumed_items_count.keys(), consumed_items_count.values()))
            df_neighbors.columns = ['item_id', 'score']
            df_neighbors = df_neighbors.sort_values(by='score', ascending=False)
            
            if len(l_prod)>1:
                recommendations=pd.concat([recommendations,df_neighbors[1:2]])
            else:
                recommendations=pd.concat([recommendations,df_neighbors[1:6]])

    return recommendations.head(n)

def rp_iknn(df:pd.DataFrame,df_f:pd.DataFrame,l_prod:list, user_id, n:int):
    try:
        df_k=df.reset_index()
        df_k['produto_full']=df_k['categoria']+" "+df_k['tipo_categoria']+" "+df_k['produto']+" "+df_k['prodcomplemento']
        df_k['produto_f']=df_k['produto']+" "+df_k['prodcomplemento']
        df_k['timestamp']=pd.to_datetime(df_k.dth_agendamento).map(pd.Timestamp.timestamp)
        df_k=df_k[['produto_full','cliente_nome','produto_f','timestamp']].groupby(['produto_f','cliente_nome','timestamp']).count()
        df_k.reset_index(inplace=True)
        encoder=MinMaxScaler(feature_range=(1, df_k.produto_full.unique()[-1]))
        df_k['rating']=pd.DataFrame(encoder.fit_transform(df_k.produto_full.array.reshape(-1, 1)))

        df_kr=pd.DataFrame()
        df_kr['userID']=df_k['cliente_nome']
        df_kr['itemID']=df_k['produto_f']
        df_kr['rating']=df_k['rating']
        df_kr['timestamp']=df_k['timestamp']

        reader = Reader(rating_scale=(1, df_k.produto_full.unique()[-1]))

        train_size = 0.8
        # Ordenar por timestamp
        df_kr = df_kr.sort_values(by='timestamp', ascending=True)

        # Definindo train e valid sets
        df_train_set, df_valid_set = np.split(df_kr, [ int(train_size*df_kr.shape[0]) ])

        train_set = (
            Dataset
            .load_from_df(df_train_set[['userID', 'itemID', 'rating']], reader)
            .build_full_trainset()
        )

        sim_options = {
        "name": "pearson_baseline",
        "user_based": False,  # compute similarities between items
        }
        model = KNNWithMeans(k=40, sim_options=sim_options, verbose=True)
        model.fit(train_set)
        
        df_predictions = pd.DataFrame(columns=['item_id', 'score'])
        for item_id in df_k.produto_f.values:
            prediction = model.predict(uid=user_id, iid=item_id).est
            df_predictions.loc[df_predictions.shape[0]] = [item_id, prediction]
    
        recommendations = (
            df_predictions
            .sort_values(by='score', ascending=False)
            .set_index('item_id')
            )
    except (IndexError,ValueError) as e:
        df_k=df_f.reset_index()
        df_k['produto_full']=df_k['categoria']+" "+df_k['tipo_categoria']+" "+df_k['produto']+" "+df_k['prodcomplemento']
        df_k['produto_f']=df_k['produto']+" "+df_k['prodcomplemento']
        df_k['timestamp']=pd.to_datetime(df_k.dth_agendamento).map(pd.Timestamp.timestamp)
        df_k=df_k[['produto_full','cliente_nome','produto_f','timestamp']].groupby(['produto_f','cliente_nome','timestamp']).count()
        df_k.reset_index(inplace=True)
        encoder=MinMaxScaler(feature_range=(1, df_k.produto_full.unique()[-1]))
        df_k['rating']=pd.DataFrame(encoder.fit_transform(df_k.produto_full.array.reshape(-1, 1)))

        df_kr=pd.DataFrame()
        df_kr['userID']=df_k['cliente_nome']
        df_kr['itemID']=df_k['produto_f']
        df_kr['rating']=df_k['rating']
        df_kr['timestamp']=df_k['timestamp']

        reader = Reader(rating_scale=(1, df_k.produto_full.unique()[-1]))

        train_size = 0.8
        # Ordenar por timestamp
        df_kr = df_kr.sort_values(by='timestamp', ascending=True)

        # Definindo train e valid sets
        df_train_set, df_valid_set = np.split(df_kr, [ int(train_size*df_kr.shape[0]) ])

        train_set = (
            Dataset
            .load_from_df(df_train_set[['userID', 'itemID', 'rating']], reader)
            .build_full_trainset()
        )

        sim_options = {
        "name": "pearson_baseline",
        "user_based": False,  # compute similarities between items
        }
        model = KNNWithMeans(k=40, sim_options=sim_options, verbose=True)
        model.fit(train_set)
        
        df_predictions = pd.DataFrame(columns=['item_id', 'score'])
        for item_id in df_k.produto_f.values:
            prediction = model.predict(uid=user_id, iid=item_id).est
            df_predictions.loc[df_predictions.shape[0]] = [item_id, prediction]
    
        recommendations = (
            df_predictions
            .sort_values(by='score', ascending=False)
            .set_index('item_id')
            )


    return recommendations.head(n)

def rp_fsvd(df:pd.DataFrame,df_f:pd.DataFrame,l_prod:list,user_id,n:int):
    try:
        df_svd=df.copy()
        df_svd=df.reset_index()
        df_svd['produto_full']=df_svd['categoria']+" "+df_svd['tipo_categoria']+" "+df_svd['produto']+" "+df_svd['prodcomplemento']
        df_svd['produto_f']=df_svd['produto']+" "+df_svd['prodcomplemento']
        df_svd['timestamp']=pd.to_datetime(df_svd.dth_agendamento).map(pd.Timestamp.timestamp)

        df_svd_r=df_svd[['produto_full','produto_f']].groupby(['produto_full']).count()
        df_svd_r.reset_index(inplace=True)
        df_svd_r.rename({'produto_f':'rating'}, axis=1,inplace=True)

        df_svd=df_svd[['produto_full','cliente_nome','produto_f','timestamp']].merge(df_svd_r[['rating']], left_index=True, right_index=True)
        #encoder=MinMaxScaler(feature_range=(1, df_svd.produto_f.unique()[-1]))
        encoder=MinMaxScaler(feature_range=(1, 5))
        df_svd['rating']=pd.DataFrame(encoder.fit_transform(df_svd.rating.array.reshape(-1, 1)))

        train_size = 0.8
        # Ordenar por timestamp
        df_svd.sort_values(by='timestamp', ascending=True, inplace=True)

        df_svd.rename(columns={'cliente_nome': 'u_id', 'produto_f': 'i_id'},inplace=True)
        

        # Definindo train e valid sets
        df_train_set, df_valid_set = np.split(df_svd, [ int(train_size*df_svd.shape[0]) ])
        df_valid_set, df_test_set = np.split(df_valid_set, [ int(0.5*df_valid_set.shape[0]) ])

        model = SVD(
        lr=0.001, # Learning rate.
        reg=0.005, # L2 regularization factor.
        n_epochs=100, # Number of SGD iterations.
        n_factors=30, # Number of latent factors.
        early_stopping=True, # Whether or not to stop training based on a validation monitoring.
        min_delta=0.0001, # Minimun delta to argue for an improvement.
        shuffle=False, # Whether or not to shuffle the training set before each epoch.
        min_rating=1, # Minimum value a rating should be clipped to at inference time.
        max_rating=5 # Maximum value a rating should be clipped to at inference time.
        )
        model.fit(X=df_train_set, X_val=df_valid_set)
        #df_valid_set['prediction'] = model.predict(df_test_set)

        #item_ids = df_valid_set['i_id'].unique()
        item_ids = df_svd['i_id'].unique()
    except (ValueError) as e:
        df_svd=df_f.copy()
        df_svd=df_f.reset_index()
        df_svd.reset_index()
        df_svd['produto_full']=df_svd['categoria']+" "+df_svd['tipo_categoria']+" "+df_svd['produto']+" "+df_svd['prodcomplemento']
        df_svd['produto_f']=df_svd['produto']+" "+df_svd['prodcomplemento']
        df_svd['timestamp']=pd.to_datetime(df_svd.dth_agendamento).map(pd.Timestamp.timestamp)

        df_svd_r=df_svd[['produto_full','produto_f']].groupby(['produto_full']).count()
        df_svd_r.reset_index(inplace=True)
        df_svd_r.rename({'produto_f':'rating'}, axis=1,inplace=True)

        df_svd=df_svd[['produto_full','cliente_nome','produto_f','timestamp']].merge(df_svd_r[['rating']], left_index=True, right_index=True)
        #encoder=MinMaxScaler(feature_range=(1, df_svd.produto_f.unique()[-1]))
        encoder=MinMaxScaler(feature_range=(1, 5))
        df_svd['rating']=pd.DataFrame(encoder.fit_transform(df_svd.rating.array.reshape(-1, 1)))

        train_size = 0.8
        # Ordenar por timestamp
        df_svd.sort_values(by='timestamp', ascending=True, inplace=True)

        df_svd.rename(columns={'cliente_nome': 'u_id', 'produto_f': 'i_id'},inplace=True)
        

        # Definindo train e valid sets
        df_train_set, df_valid_set = np.split(df_svd, [ int(train_size*df_svd.shape[0]) ])
        df_valid_set, df_test_set = np.split(df_valid_set, [ int(0.5*df_valid_set.shape[0]) ])

        model = SVD(
        lr=0.001, # Learning rate.
        reg=0.005, # L2 regularization factor.
        n_epochs=100, # Number of SGD iterations.
        n_factors=30, # Number of latent factors.
        early_stopping=True, # Whether or not to stop training based on a validation monitoring.
        min_delta=0.0001, # Minimun delta to argue for an improvement.
        shuffle=False, # Whether or not to shuffle the training set before each epoch.
        min_rating=1, # Minimum value a rating should be clipped to at inference time.
        max_rating=5 # Maximum value a rating should be clipped to at inference time.
        )
        model.fit(X=df_train_set, X_val=df_valid_set)
        #df_valid_set['prediction'] = model.predict(df_test_set)

        #item_ids = df_valid_set['i_id'].unique()
        item_ids = df_svd['i_id'].unique()
        
    df_predictions = pd.DataFrame()
    df_predictions['i_id'] = item_ids
    df_predictions['u_id'] = user_id
    df_predictions['score'] = model.predict(df_predictions)
    df_predictions.sort_values(by='score', ascending=False,inplace=True)
    df_predictions.rename(columns={'i_id': 'item_id'}, inplace=True)
    df_predictions.set_index('item_id',inplace=True)
    recommendations=df_predictions

    return recommendations.head(n)

def rp_lfm(df:pd.DataFrame,df_f:pd.DataFrame,user_id,n:int):
    """based on:
        https://towardsdatascience.com/how-i-would-explain-building-lightfm-hybrid-recommenders-to-a-5-year-old-b6ee18571309
    """
    try:
        dfl=df.reset_index()
        df_l=dfl[['cod_pedido','produto_f']].groupby('cod_pedido').agg({'produto_f': lambda x : ','.join(set(x))})
        df_l.rename(columns={'produto_f':'itens'},inplace=True)
        df_l.reset_index(inplace=True)
        df_l['itens']=df_l.itens.str.split(pat=',')
        df_l.head()

        #preparando os dados:
        df_lf=df.reset_index()
        df_lfdm=pd.get_dummies(df_lf[['categoria','tipo_categoria','produto','prodcomplemento']])#features
        df_lf['produto_full']=df_lf['categoria']+" "+df_lf['tipo_categoria']+" "+df_lf['produto']+" "+df_lf['prodcomplemento']
        df_lf['produto_f']=df_lf['produto']+" "+df_lf['prodcomplemento']
        df_lf['timestamp']=pd.to_datetime(df_lf.dth_agendamento).map(pd.Timestamp.timestamp)
        df_lfc=df_lf[['cliente_nome']].merge(df_lf[['produto_f']],left_index=True,right_index=True)
        df_lf=df_lfc.merge(df_lfdm,left_index=True, right_index=True)
        df_lf=df_lf.groupby(['cliente_nome','produto_f']).sum()
        df_lf.reset_index(inplace=True)

        train_size = 0.8
        # Definindo train e valid sets
        train_lfm, test_lfm= np.split(df_lf, [ int(train_size*df_lf.shape[0]) ])

        item_f = []
        col=[]

        unique_f = []
        for i in df_lfdm.columns.to_list():
            counter=0
            while counter < len(df_lfdm[i].unique()):
                col.append(i)
                counter+=1
            for j in df_lfdm[i].unique():
                unique_f.append(j)  
        #print('f1:', unique_f1)
        for x,y in zip(col, unique_f):
            res = str(x)+ ":" +str(y)
            item_f.append(res)

        #creating the lfm dataset:
        dataset = DT(user_identity_features=True, item_identity_features=False)
        dataset.fit(train_lfm.cliente_nome.unique(),train_lfm.produto_f.unique(),item_features =item_f)
        interactions, weights=dataset.build_interactions([(x[0], x[1]) for x in train_lfm.values])

        

        user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
        model_lfm = LightFM(loss='warp')

        # predict for existing user
        model_lfm.fit(interactions, user_features=None, item_features=None, sample_weight=None, epochs=10, num_threads=1, verbose=False)
        user_x = user_id_map[user_id]
        n_users, n_items = interactions.shape # no of users * no of items
        score=model_lfm.predict(user_x, np.arange(n_items)) # means predict for all 

    except (KeyError) as e:
        dfl=df_f.reset_index()
        df_l=dfl[['cod_pedido','produto_f']].groupby('cod_pedido').agg({'produto_f': lambda x : ','.join(set(x))})
        df_l.rename(columns={'produto_f':'itens'},inplace=True)
        df_l.reset_index(inplace=True)
        df_l['itens']=df_l.itens.str.split(pat=',')
        df_l.head()

        #preparando os dados:
        df_lf=df_f.reset_index()
        df_lfdm=pd.get_dummies(df_lf[['categoria','tipo_categoria','produto','prodcomplemento']])#features
        df_lf['produto_full']=df_lf['categoria']+" "+df_lf['tipo_categoria']+" "+df_lf['produto']+" "+df_lf['prodcomplemento']
        df_lf['produto_f']=df_lf['produto']+" "+df_lf['prodcomplemento']
        df_lf['timestamp']=pd.to_datetime(df_lf.dth_agendamento).map(pd.Timestamp.timestamp)
        df_lfc=df_lf[['cliente_nome']].merge(df_lf[['produto_f']],left_index=True,right_index=True)
        df_lf=df_lfc.merge(df_lfdm,left_index=True, right_index=True)
        df_lf=df_lf.groupby(['cliente_nome','produto_f']).sum()
        df_lf.reset_index(inplace=True)

        train_size = 0.8
        # Definindo train e valid sets
        train_lfm, test_lfm= np.split(df_lf, [ int(train_size*df_lf.shape[0]) ])

        item_f = []
        col=[]

        unique_f = []
        for i in df_lfdm.columns.to_list():
            counter=0
            while counter < len(df_lfdm[i].unique()):
                col.append(i)
                counter+=1
            for j in df_lfdm[i].unique():
                unique_f.append(j)  
        #print('f1:', unique_f1)
        for x,y in zip(col, unique_f):
            res = str(x)+ ":" +str(y)
            item_f.append(res)

        #creating the lfm dataset:
        dataset = DT(user_identity_features=True, item_identity_features=False)
        dataset.fit(train_lfm.cliente_nome.unique(),train_lfm.produto_f.unique(),item_features =item_f)
        interactions, weights=dataset.build_interactions([(x[0], x[1]) for x in train_lfm.values])

        user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
        model_lfm = LightFM(loss='warp')

        # predict for existing user
        model_lfm.fit(interactions, user_features=None, item_features=None, sample_weight=None, epochs=10, num_threads=1, verbose=False)
        user_x = user_id_map[user_id]
        n_users, n_items = interactions.shape # no of users * no of items
        score=model_lfm.predict(user_x, np.arange(n_items)) # means predict for all 

    recommendations=pd.DataFrame()
    recommendations['items']=item_id_map.keys()
    recommendations['score']=score
    recommendations.set_index('items',inplace=True)
    recommendations.sort_values(by='score',ascending=False,inplace=True)

    return recommendations.head(n)  


def r_np(df_loja_rec,l_prod,n,h):
    df_loja_rec1=df_loja_rec.copy()
    df_loja_rec1['dth_hora']=df_loja_rec1['dth_agendamento'].apply(extract_hour)
    df_loja_recnp=time_filter(df_loja_rec1,hr=h,nh=1)
    #st.write ('Quantidade de linhas apos antes do filtro de horario',df_loja_rec.shape[0])
    #st.write ('Quantidade de linhas apos o filtro de horario',df_loja_recnp.shape[0])
    if len(l_prod)==0:
        placeholder1 = st.empty() 
    else:
        tab1, tab2, tab3 = st.tabs(["Apriori", "Top N","Content Based"])
        with tab1:          
            rec_np=rnp_apr(df_loja_recnp,l_prod,n)
            placeholder1 = st.empty()
            placeholder1.text("Quem comprou estes produtos também comprou:")
            if rec_np.shape[0]>0:
                with placeholder1.container():
                    if len(l_prod)>1:
                        st.write("Quem comprou estes produtos também comprou:")
                        for i in rec_np.consequents:
                            st.write(i)
                    else:
                        st.write("Quem comprou este produto também comprou:")
                        for i in rec_np.consequents:
                            st.write(i)  
            else:
                with placeholder1.container():
                        st.write("Sem proposições para este item")
        with tab2:
            rec_np=rnp_top_n(df_loja_recnp,l_prod=l_prod,n=n)
            placeholder1 = st.empty()
            placeholder1.text("Adicione ao carrinho os produtos mais vendidos:")
            with placeholder1.container():
                    st.write("Adicione ao carrinho os produtos mais vendidos:")
                    for i in rec_np.produto_f:
                        st.write(i)
        with tab3:
            rec_np=rnp_cb(df_loja_recnp,df_loja_rec1,l_prod,n)
            placeholder1 = st.empty()
            placeholder1.text("Quem comprou estes produtos também comprou:")
            with placeholder1.container():
                    if len(l_prod)>1:
                        st.write("Quem comprou estes produtos também comprou:")
                        for i in rec_np.produto_f:
                            st.write(i)
                    else:
                        st.write("Quem comprou este produto também comprou:")
                        for i in rec_np.produto_f:
                            st.write(i)            
    return df_loja_rec.shape[0], df_loja_recnp.shape[0]

def r_p(df_loja_rec,l_prod,user_id,n,h):
    df_loja_rec1=df_loja_rec.copy()
    df_loja_rec1['dth_hora']=df_loja_rec1['dth_agendamento'].apply(extract_hour)
    df_loja_recnp=time_filter(df_loja_rec1,hr=h,nh=1)
    if len(l_prod)==0:
        placeholder2 = st.empty() 
    else:
        tab4, tab5, tab6, tab7 = st.tabs(["Co-visitation", 'Item KNN','Funk-SVD','LightFM'])
        with tab4:          
            rec_p=rp_cv(df_loja_recnp,df_loja_rec1,l_prod,n)
            placeholder2 = st.empty()
            placeholder2.text("Quem comprou estes produtos também comprou:")
            with placeholder2.container():
                    if len(l_prod)>1:
                        st.write("Quem comprou estes produtos também comprou:")
                        for i in rec_p.item_id:
                            st.write(i)
                    else:
                        st.write("Quem comprou este produto também comprou:")
                        for i in rec_p.item_id:
                            st.write(i)
        with tab5: 
            rec_p=rp_iknn(df_loja_recnp,df_loja_rec1,l_prod,user_id,n)
            placeholder2 = st.empty()
            placeholder2.text("Quem comprou estes produtos também comprou:")
            with placeholder2.container():
                    if len(l_prod)>1:
                        st.write("Quem comprou estes produtos também comprou:")
                        for i in rec_p.index:
                            st.write(i)
                    else:
                        st.write("Quem comprou este produto também comprou:")
                        for i in rec_p.index:
                            st.write(i)
        with tab6:
            rec_p=rp_fsvd(df_loja_recnp,df_loja_rec1,l_prod,user_id,n)
            placeholder2 = st.empty()
            placeholder2.text("Quem comprou estes produtos também comprou:")
            with placeholder2.container():
                    if len(l_prod)>1:
                        st.write("Quem comprou estes produtos também comprou:")
                        for i in rec_p.index:
                            st.write(i)
                    else:
                        st.write("Quem comprou este produto também comprou:")
                        for i in rec_p.index:
                            st.write(i)
        with tab7:
            rec_p=rp_lfm(df_loja_recnp,df_loja_rec1,user_id,n)
            placeholder2 = st.empty()
            placeholder2.text("Quem comprou estes produtos também comprou:")
            with placeholder2.container():
                    if len(l_prod)>1:
                        st.write("Quem comprou estes produtos também comprou:")
                        for i in rec_p.index:
                            st.write(i)
                    else:
                        st.write("Quem comprou este produto também comprou:")
                        for i in rec_p.index:
                            st.write(i)

##############  METRICAS #################################
def master_m(df_items):
    filepath = './valid_metrics.parquet'
    df = pd.read_parquet(filepath)

    RANKS = list(range(1, 21))

    item_ids = df_items.produto_f.unique().tolist()
    coverage_report = get_coverage_report(df, RANKS, item_ids)
    ranking_report = get_ranking_report(df, RANKS)
    classification_report = get_classification_report(df, RANKS)


    return coverage_report, ranking_report, classification_report

def convert_coverage_metrics(df, rank=20):
  """recommended items"""
  df_true = df
  df_true = df_true.explode('y_true')[['model', 'user_id', 'y_true']]
  df_true['item_id'] = df_true['y_true'].apply(lambda x: x.get('item_id'))
  df_true['y_true'] = df_true['y_true'].apply(lambda x: x.get('rating'))

  df_score = df
  df_score = df_score.explode('y_score')[['model', 'user_id', 'y_score']]
  df_score['item_id'] = df_score['y_score'].apply(lambda x: x.get('item_id'))
  df_score['y_score'] = df_score['y_score'].apply(lambda x: x.get('score'))
  df_score['y_score'] = df_true['y_true'].max() * df_score['y_score'] / df_score['y_score'].max()
  
  df_metrics = df_true.merge(df_score, on=['model', 'user_id', 'item_id'], how='outer')
  df_metrics.sort_values(by=['user_id', 'y_score'], ascending=False, inplace=True)
  df_metrics['rank'] = df_metrics.groupby(['model', 'user_id'])['y_score'].rank(method='first', ascending=False)
  df_metrics = df_metrics.query('rank <= @rank')
  df_metrics['y_true'] = df_metrics['y_true'].fillna(0)

  return df_metrics[['model', 'user_id', 'item_id', 'y_true', 'y_score']].reset_index(drop=True)

def convert_ranking_metrics(df, rank=20):
  """
    y_true: recommended items that were consumed
    y_score: recommended items within rank
  """
  df_true = df
  df_true = df_true.explode('y_true')[['model', 'user_id', 'y_true']]
  df_true['item_id'] = df_true['y_true'].apply(lambda x: x.get('item_id'))
  df_true['y_true'] = df_true['y_true'].apply(lambda x: x.get('rating'))
  df_true = df_true.query('y_true > 0')
  df_true = df_true.groupby(['model', 'user_id']).agg({'item_id': list}).reset_index()
  df_true.rename({'item_id': 'y_true'}, axis=1, inplace=True)

  df_score = df
  df_score = df_score.explode('y_score')[['model', 'user_id', 'y_score']]
  df_score['item_id'] = df_score['y_score'].apply(lambda x: x.get('item_id'))
  df_score['y_score'] = df_score['y_score'].apply(lambda x: x.get('score'))
  df_score['rank'] = df_score.groupby(['model', 'user_id'])['y_score'].rank(method='first', ascending=False)
  df_score = df_score.query('rank <= @rank')
  df_score = df_score.groupby(['model', 'user_id']).agg({'item_id': list}).reset_index()
  df_score.rename({'item_id': 'y_score'}, axis=1, inplace=True)
  
  df_metrics = df_score.merge(df_true, on=['model', 'user_id'], how='left')
  df_metrics['y_true'] = df_metrics.apply(lambda x: list(set(x['y_true']).intersection(x['y_score'])), axis=1)
  
  return df_metrics[['model', 'user_id', 'y_true', 'y_score']].reset_index(drop=True)

def convert_classification_metrics(df, threshold=1, rank=20):
  """known or recommended items"""
  df_score = df
  df_score = df_score.explode('y_score')[['model', 'user_id', 'y_score']]
  df_score['item_id'] = df_score['y_score'].apply(lambda x: x.get('item_id'))
  df_score['y_score'] = df_score['y_score'].apply(lambda x: x.get('score'))
  
  # Normalize
  df_score['y_score'] = df_score['y_score']/df_score['y_score'].max()

  df_true = df
  df_true = df_true.explode('y_true')[['model', 'user_id', 'y_true']]
  df_true['item_id'] = df_true['y_true'].apply(lambda x: x.get('item_id'))
  df_true['y_true'] = df_true['y_true'].apply(lambda x: x.get('rating'))
  
  df_metrics = df_true.merge(df_score, on=['model', 'user_id', 'item_id'], how='outer')
  df_metrics.sort_values(by=['user_id', 'y_score'], ascending=False, inplace=True)
  df_metrics['rank'] = df_metrics.groupby(['model', 'user_id'])['y_score'].rank(ascending=False)
  df_metrics['y_score'] = df_metrics.apply(lambda x: x['y_score'] if x['rank'] <= rank else 0, axis=1) 
  df_metrics['y_true'] = (df_metrics['y_true'] >= threshold).astype(int)

  return df_metrics[['model', 'user_id', 'item_id', 'y_true', 'y_score']].reset_index(drop=True)

def get_coverage_report(df, ranks, item_ids):
  coverage_report = pd.DataFrame(columns=['model', 'rank', 'item_coverage'])
  for rank in ranks:
    for i, model in enumerate(df['model'].unique()):
      df_metrics = convert_coverage_metrics(
          df.query('model == @model'),
          rank=rank
      )
      user_ids = df_metrics['user_id'].unique().tolist()
      user_items = df_metrics[['user_id', 'item_id']].values.tolist()
      coverage = item_coverage((user_ids, item_ids), user_items)

      coverage_report.loc[coverage_report.shape[0]] = [model, rank, coverage]

  return coverage_report.sort_values(by=['model', 'rank']).reset_index(drop=True)

def get_ranking_report(df, ranks):
  ranking_report = pd.DataFrame(columns=['model', 'rank', 'mrr', 'personalization'])
  for rank in ranks:
    for i, model in enumerate(df['model'].unique()):
      df_metrics = convert_ranking_metrics(
          df.query('model == @model'),
          rank=rank
      )

      mrr = df_metrics.apply(
          lambda x: mean_reciprocal_rank(x["y_true"], x["y_score"]) if len(x['y_true']) > 0 and len(x['y_score']) else 0,
          axis=1
      ).mean()
      pers = personalization(df_metrics['y_score'])
      ranking_report.loc[ranking_report.shape[0]] = [model, rank, mrr, pers]

  return ranking_report.sort_values(by=['model', 'rank']).reset_index(drop=True)

def get_classification_report(df, ranks):
  """Classification report for each rank and model"""
  classification_report = pd.DataFrame(columns=['model', 'rank', 'precision', 'recall'])
  for rank in ranks:
    for i, model in enumerate(df['model'].unique()):
      df_metrics = convert_classification_metrics(
          df.query('model == @model'),
          threshold=1,
          rank=rank
      )

      precision = precision_score(df_metrics['y_true'], df_metrics['y_score'] >= 0.5)
      recall = recall_score(df_metrics['y_true'], df_metrics['y_score'] >= 0.5)

      classification_report.loc[classification_report.shape[0]] = [model, rank, precision, recall]

  return classification_report.sort_values(by=['model', 'rank']).reset_index(drop=True)


def plot_report(report, figsize=(16,10)):
  metrics = report.drop(['model', 'rank'], axis=1).columns
  fig, axes = plt.subplots(nrows=len(metrics), ncols=1, sharex=True, figsize=figsize)
  axes = [axes] if len(metrics) == 1 else axes
  for i, metric in enumerate(metrics):
    ax = axes[i]
    for model in report['model'].unique():
      df_plot = report.query('model == @model').sort_values(by='rank')
      ax.plot(df_plot['rank'], df_plot[metric], label=model)
      ax.scatter(df_plot['rank'], df_plot[metric])

    ax.set_xticks([int(rank) for rank in df_plot['rank']])
    ax.set_title(metric.title())
    ax.legend()
    ax.grid(True, linestyle='--')

  ax.set_xlabel('Rank')
  return st.pyplot(fig)


def m_topn(df_f):
    
    recommendations = (
        df_f
        .groupby('produto_f')
        .count()['cliente_nome']
        .reset_index()
        .rename({'cliente_nome': 'score'}, axis=1)
        .sort_values(by='score', ascending=False)
    )

    train_size = 0.8
    df_f.sort_values(by='dth_hora', inplace=True)
    df_train_set, df_valid_set= np.split(df_f, [int(train_size * df_f.shape[0])])

    recommendations = rnp_top_n(df_f, l_prod=None, n=20)
    scores = [{'item_id': x['produto_f'], 'score': x['score']} for _, x in recommendations.iterrows()]

    model_name = 'top'
    df_predictions = df_valid_set
    df_predictions['y_true'] = df_predictions.apply(lambda x: {'item_id': x['item_id'], 'rating': x['rating']}, axis=1)
    df_predictions = df_predictions.groupby('user_id').agg({'y_true': list})
    df_predictions['y_score'] = df_predictions.apply(lambda x: scores, axis=1)
    df_predictions['model'] = model_name
    df_predictions.reset_index(drop=False, inplace=True)

    column_order = ['model', 'user_id', 'y_true', 'y_score']
    df_predictions[column_order].to_parquet(f'valid_{model_name}.parquet', index=None)

    return

def m_iknn(df_f):
    model_name = 'itemknn'
    n= 20

    df_k=df_f.reset_index()
    df_k['produto_full']=df_k['categoria']+" "+df_k['tipo_categoria']+" "+df_k['produto']+" "+df_k['prodcomplemento']
    df_k['produto_f']=df_k['produto']+" "+df_k['prodcomplemento']
    df_k['timestamp']=pd.to_datetime(df_k.dth_agendamento).map(pd.Timestamp.timestamp)
    df_k=df_k[['produto_full','cliente_nome','produto_f','timestamp']].groupby(['produto_f','cliente_nome','timestamp']).count()
    df_k.reset_index(inplace=True)
    encoder=MinMaxScaler(feature_range=(1, df_k.produto_full.unique()[-1]))
    df_k['rating']=pd.DataFrame(encoder.fit_transform(df_k.produto_full.array.reshape(-1, 1)))

    df_kr=pd.DataFrame()
    df_kr['userID']=df_k['cliente_nome']
    df_kr['itemID']=df_k['produto_f']
    df_kr['rating']=df_k['rating']
    df_kr['timestamp']=df_k['timestamp']

    reader = Reader(rating_scale=(1, df_k.produto_full.unique()[-1]))

    train_size = 0.8
    # Ordenar por timestamp
    df_kr = df_kr.sort_values(by='timestamp', ascending=True)

    # Definindo train e valid sets
    df_train_set, df_valid_set = np.split(df_kr, [ int(train_size*df_kr.shape[0]) ])


    df_recommendations = pd.DataFrame()
    catalog = df_k.produto_full.values
    for user_id in df_valid_set['userID'].unique():
        user_known_items = df_train_set.query('userID == @user_id')['itemID'].unique()
        recommendable_items = np.array(list(set(catalog)-set(user_known_items)))
        user_recommendations = rp_iknn(df_f,df_f,l_prod=None, user_id=user_id,n=n).reset_index(drop=False)
        user_recommendations['user_id'] = user_id
        df_recommendations = pd.concat([df_recommendations, user_recommendations])

    df_recommendations['model'] = model_name
    df_recommendations = df_recommendations.merge(
        df_valid_set,
        left_on=['user_id', 'item_id'],
        right_on=['userID', 'itemID'],
        how='left'
    )
    df_recommendations = df_recommendations[['model', 'user_id', 'item_id', 'rating', 'score']]
    df_recommendations['rating'] = df_recommendations['rating'].fillna(0)
    df_rec_bkp = df_recommendations.copy()
    
    df_recommendations['y_score'] = df_recommendations.apply(lambda x: {'item_id': x['item_id'], 'score': x['score']}, axis=1)
    df_recommendations = df_recommendations.groupby(['model', 'user_id']).agg({'y_score': list}).reset_index(drop=False)

    df_predictions = df_valid_set.rename({'userID': 'user_id', 'itemID': 'item_id'}, axis=1)
    df_predictions['y_true'] = df_predictions.apply(lambda x: {'item_id': x['item_id'], 'rating': x['rating']}, axis=1)
    df_predictions = df_predictions.groupby('user_id').agg({'y_true': list}).reset_index(drop=False)
    df_predictions = df_predictions.merge(df_recommendations, on='user_id', how='inner')

    df_predictions.to_parquet(f'valid_{model_name}.parquet', index=None)
    return


def m_svd(df_f,user_id):
    model_name = 'svd'
    n = 20
    df_svd=df_f.copy()
    df_svd=df_f.reset_index()
    df_svd.reset_index()
    df_svd['produto_full']=df_svd['categoria']+" "+df_svd['tipo_categoria']+" "+df_svd['produto']+" "+df_svd['prodcomplemento']
    df_svd['produto_f']=df_svd['produto']+" "+df_svd['prodcomplemento']
    df_svd['timestamp']=pd.to_datetime(df_svd.dth_agendamento).map(pd.Timestamp.timestamp)

    df_svd_r=df_svd[['produto_full','produto_f']].groupby(['produto_full']).count()
    df_svd_r.reset_index(inplace=True)
    df_svd_r.rename({'produto_f':'rating'}, axis=1,inplace=True)

    df_svd=df_svd[['produto_full','cliente_nome','produto_f','timestamp']].merge(df_svd_r[['rating']], left_index=True, right_index=True)
    #encoder=MinMaxScaler(feature_range=(1, df_svd.produto_f.unique()[-1]))
    encoder=MinMaxScaler(feature_range=(1, 5))
    df_svd['rating']=pd.DataFrame(encoder.fit_transform(df_svd.rating.array.reshape(-1, 1)))

    # Ordenar por timestamp
    df_svd.sort_values(by='timestamp', ascending=True, inplace=True)

    df_svd.rename(columns={'cliente_nome': 'u_id', 'produto_f': 'i_id'},inplace=True)
        
    # Definindo train e valid sets
    train_size=0.8
    df_train_set, df_valid_set = np.split(df_svd, [ int(train_size*df_svd.shape[0]) ])
    catalog = df_svd.produto_full.values
    df_recommendations = pd.DataFrame()
    for user_id in df_valid_set['u_id'].unique():
        user_known_items = df_train_set.query('u_id == @user_id')['i_id'].unique()
        recommendable_items = np.array(list(set(catalog)-set(user_known_items)))
        
        user_recommendations = rp_fsvd(df_f,df_f,l_prod=recommendable_items,user_id=user_id,n=n).reset_index(drop=False)
        user_recommendations['user_id'] = user_id
        df_recommendations = pd.concat([df_recommendations, user_recommendations])

    df_recommendations['y_score'] = df_recommendations.apply(lambda x: {'item_id': x['item_id'], 'score': x['score']}, axis=1)
    df_recommendations = df_recommendations.groupby('user_id').agg({'y_score': list}).reset_index(drop=False)

    df_predictions = df_valid_set.rename({'u_id': 'user_id', 'i_id': 'item_id'}, axis=1)
    df_predictions['y_true'] = df_predictions.apply(lambda x: {'item_id': x['item_id'], 'rating': x['rating']}, axis=1)
    df_predictions = df_predictions.groupby('user_id').agg({'y_true': list}).reset_index(drop=False)
    df_predictions = df_predictions.merge(df_recommendations, on='user_id', how='inner')
    df_predictions['model'] = model_name
    
    column_order = ['model', 'user_id', 'y_true', 'y_score']
    df_predictions[column_order].to_parquet(f'valid_{model_name}.parquet', index=None)

    return
