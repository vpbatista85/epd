import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import mlxtend
import os
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


from funk_svd import SVD
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
from streamlit.source_util import (
    page_icon_and_name, 
    calc_md5, 
    get_pages,
    _on_pages_changed
)




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

st.cache(suppress_st_warning=True)
def f_escolha(df):

    delete_page("C:\Users\vitor\Documents\Python\streamlit\Scripts\teste_strealit_main.py", "shop")
    delete_page("C:\Users\vitor\Documents\Python\streamlit\Scripts\teste_strealit_main.py", "cart")

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
        add_page('C:\Users\vitor\Documents\Python\streamlit\Scripts\teste_strealit_main.py', 'shop')

    ##Seleção dos campos referente ao produto:
    st.write('Selecione o produto para o carrinho:')
    df_loja=df[df['loja_compra']==store]
    df_loja_recnp=df_loja.copy()
    df_loja_recnp['produto_f']=df_loja_recnp['produto']+" "+df_loja_recnp['prodcomplemento']

    #Seleção da categoria do produto
    cat = st.selectbox(
        'Selecione a categoria:',
        df_loja.categoria.unique())
    df_cat=df_loja[df_loja['categoria']==cat]

    #Seleção do tipo do produto
    tipo = st.selectbox(
        'Selecione o tipo:',
        df_cat.tipo_categoria.unique())
    df_tipo=df_cat[df_cat['tipo_categoria']==tipo]
    #Seleção do produto
    product=st.selectbox(
        'Selecione o produto:',
        df_tipo.produto.unique())
    df_prod=df_tipo[df_tipo['produto']==product]
    #Seleção do complemento

    if df_prod.prodcomplemento.isin([""]).count()>=1 and len(df_prod.prodcomplemento.unique())==1:
            p_dis=True
            p_vis="collapsed"
    else:
        p_dis=False
        p_vis="visible"

    complement=st.selectbox(
        'Selecione o complemento:',
        df_prod.prodcomplemento.unique(),
        disabled=p_dis,
        label_visibility=p_vis)

    df_compl=df_prod[df_prod['prodcomplemento']==complement]

    prodf=product+" "+str(complement)


    if st.button('Add carrinho'):
            st.write(prodf,"adicionado ao carrinho.")
            st.session_state.l_prod.append(prodf)

    if len(st.session_state.l_prod)==0:
        state=True
    else:
        state=False
    if st.button('Del item',disabled=state):
        if len(st.session_state.l_prod)==1:
            st.write (f"{st.session_state.l_prod[-1]}, removido do carrinho.")
            st.session_state.l_prod=[]
            placeholder.empty()
        else:    
            st.write (f"{st.session_state.l_prod[-1]}, removido do carrinho.")
            st.session_state.l_prod.pop()
            placeholder.empty()  
            placeholder.text("Carrinho:")
            with placeholder.container():
                st.write('Carrinho:')
                for i in st.session_state.l_prod[0:-2]:
                    st.write(i)

    if st.button('Del carrinho',disabled=state):
        st.session_state.l_prod=[]
        st.write (f"Carrinho limpo.") 
        placeholder.empty()


    return df_loja_recnp

            
def f_carrinho():
        import streamlit as st
        placeholder = st.empty()
        placeholder.text("Carrinho:")

        with placeholder.container():
            st.write('Carrinho:')
            for i in st.session_state.l_prod:
                st.write(i)

      


def rnp_apr(dfs:pd.DataFrame,l_prod):
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
   
    return recommendations

def rnp_top_n(ratings:pd.DataFrame, n:int, l_prod:list) -> pd.DataFrame:
    #Recomendação não personalizada por n produtos mais consumidos.
    recommendations = (
        ratings
        .groupby('produto_f')
        .count()['cliente_nome']
        .reset_index()
        .rename({'cliente_nome': 'score'}, axis=1)
        .sort_values(by='score', ascending=False)
    )
    for i in l_prod:
      recommendations=recommendations[recommendations.produto_f.str.contains(i, regex=False)==False]
    return recommendations.head(n)  

def rnp_cb(df:pd.DataFrame,l_prod:list)-> pd.DataFrame:
    #preparando o dataframe para aplicação do algoritimo:
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

    return recommendations.head()         

def rp_cv(df:pd.DataFrame,l_prod:list)-> pd.DataFrame:
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

    return recommendations

def r_np(df_loja_recnp,l_prod): 
    if len(l_prod)==0:
        placeholder1 = st.empty() 
    else:
        tab1, tab2, tab3 = st.tabs(["Apriori", "Top N","Content Based"])
        with tab1:          
            rec_np=rnp_apr(df_loja_recnp,l_prod)
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
            rec_np=rnp_top_n(df_loja_recnp,n=5,l_prod=l_prod)
            placeholder1 = st.empty()
            placeholder1.text("Adicione ao carrinho os produtos mais vendidos:")
            with placeholder1.container():
                    st.write("Adicione ao carrinho os produtos mais vendidos:")
                    for i in rec_np.produto_f:
                        st.write(i)
        with tab3:
            rec_np=rnp_cb(df_loja_recnp,l_prod)
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

def r_p(df_loja_recnp,l_prod):
    if len(l_prod)==0:
        placeholder2 = st.empty() 
    else:
        tab4, tab5 = st.tabs(["Co-visitation", 'Nearest Neighbors'])
        with tab4:          
            rec_p=rp_cv(df_loja_recnp,l_prod)
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


def main():

    #lista de produtos no carrinho
    #df = pd.read_csv(r"C:\Users\vitor\Documents\Python\streamlit\Scripts\output.csv", encoding = 'utf-8')
    df_server= pd.read_csv(r"https://github.com/vpbatista85/epd/blob/main/output.csv?raw=true", encoding = 'utf-8')
    df=df_server.copy()
    df.drop_duplicates(inplace=True)
    df.fillna("",inplace=True)
    df_loja_recnp=f_escolha(df)
    f_carrinho()
    r_np(df_loja_recnp,st.session_state.l_prod)
    r_p(df_loja_recnp,st.session_state.l_prod)




#inicio main:
st.set_page_config(
    page_title = 'login',
    layout ="centered",
    initial_sidebar_state ="auto"
    )

if 'l_prod' not in st.session_state:
    st.session_state.l_prod = []
if __name__ == "__main__":
     main()







   
