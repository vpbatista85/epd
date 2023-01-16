import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns


if 'l_prod' not in st.session_state:
    st.session_state.l_prod = []

def f_escolha(df,l_prod):
    
    st.title("Bem vindo!")
        #Seleção da loja 

    store = st.selectbox(
        'Selecione a Loja:',
        df['loja_compra'].unique())
    ##Seleção dos campos referente ao produto:
    st.write('Selecione o produto para o carrinho:')
    df_loja=df[df['loja_compra']==store]
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
    complement=st.selectbox(
        'Selecione o complemento:',
        df_prod.prodcomplemento.unique())
    df_compl=df_prod[df_prod['prodcomplemento']==complement]

    prodf=product+str(complement)


    if st.button('Add carrinho'):
            st.write(prodf,"adicionado ao carrinho")
            st.session_state.l_prod.append(prodf)
          
            
def f_carrinho(l_prod):
        import streamlit as st
        placeholder = st.empty()
        placeholder.text("Carrinho:")

        with placeholder.container():
            st.write('Carrinho:')
            for i in st.session_state.l_prod:
                st.write(i)
            


def main():
    #lista de produtos no carrinho
    #df = pd.read_csv(r"C:\Users\vitor\Documents\Python\streamlit\Scripts\output.csv", encoding = 'utf-8')
    df_server= pd.read_csv(r"https://github.com/vpbatista85/epd/blob/main/output.csv?raw=true", encoding = 'utf-8')
    df=df_server.copy()
    df.fillna("",inplace=True)
    f_escolha(df,l_prod)
    f_carrinho(l_prod)

#inicio main:
l_prod=[]
if __name__ == "__main__":
    main()







   
