import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def f_escolha(df):

    st.title("Bem vindo!")
        #Seleção da loja 

    store = st.selectbox(
        'Selecione a Loja:',
        df['loja_compra'].unique())
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

    prodf=product+str(complement)


    if st.button('Add carrinho'):
            st.write(prodf,"adicionado ao carrinho")
            st.session_state.l_prod.append(prodf)

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

   # n_d={}
    #for p in dfs.cod_pedido.values:
     #   a=[]
      #  t=dfs[dfs['cod_pedido']==p]
       # for i in t.produto_f.values:
        #    a.append(i)
    #n_d[p]=a

    encoder=TransactionEncoder()
    
    #te_array=encoder.fit(n_d.values()).transform(n_d.values())
    te_array=encoder.fit(list(df_l.itens)).transform(list(df_l.itens))
    dft=pd.DataFrame(data=te_array,columns=encoder.columns_)
    frequent_items=apriori(dft,min_support=0.01,use_colnames=True)
    frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))
    rules=association_rules(frequent_items, metric='confidence',min_threshold=0.3)
    rules.sort_values(by='lift',ascending=False)
    rules.antecedents=rules.antecedents.astype('string')
    rules.consequents=rules.consequents.astype('string')
    rules.antecedents=rules.antecedents.str.strip('frozenset({})')
    rules.consequents=rules.consequents.str.strip('frozenset({})')
    #recomendação
    recomendations=pd.DataFrame(columns=rules.columns)
    for i in l_prod:
        recomendations=pd.concat([recomendations,rules[rules.antecedents.str.contains(i, regex=False)]],ignore_index=True)


    return recomendations

def rnp_top_n(ratings:pd.DataFrame, n:int) -> pd.DataFrame:
    #Recomendação não personalizada por n produtos mais consumidos.
    recommendations = (
        ratings
        .groupby('produto_f')
        .count()['cliente_nome']
        .reset_index()
        .rename({'cliente_nome': 'score'}, axis=1)
        .sort_values(by='score', ascending=False)
    )

    return recommendations.head(n)           

def r_np(df_loja_recnp,l_prod): 
    if len(l_prod)==0:
        placeholder1 = st.empty() 
    else:
        tab1, tab2, = st.tabs(["Apriori", "Top N"])
        with tab1:          
            rec_np=rnp_apr(df_loja_recnp,l_prod)
            placeholder1 = st.empty()
            placeholder1.text("Quem comprou estes produtos também comprou:")
            with placeholder1.container():
                    st.write("Quem comprou estes produtos também comprou:")
                    for i in rec_np.consequents:
                        st.write(i)
        with tab2:
            rec_np=rnp_top_n(df_loja_recnp,n=5)
            placeholder1 = st.empty()
            placeholder1.text("Adicione ao carrinho os produtos mais vendidos:")
            with placeholder1.container():
                    st.write("Adicione ao carrinho os produtos mais vendidos:")
                    for i in rec_np.produto_f:
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




#inicio main:

if 'l_prod' not in st.session_state:
    st.session_state.l_prod = []
if __name__ == "__main__":
     main()







   
