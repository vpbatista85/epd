import streamlit as st
import pandas as pd
import teste_strealit_main

store=st.session_state.store
user=st.session_state.user
df=st.session_state.df
st.write(store)
st.write(user)
st.write(df)


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