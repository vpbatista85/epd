import streamlit as st
from datetime import datetime,  timedelta
# import pandas as pd
# import numpy as np
# import sklearn
# import mlxtend
# import os
# from collections import Counter
# import networkx as nx
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
import utils

if len(st.session_state.l_prod)==0:
        state=True
else:
        state=False

with st.sidebar:
    st.write('Simulação de periodo:')
    horario=st.checkbox('Horario atual', value=True, key=None, help='Marque para usar o horário local', on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    if horario:
       hora=st.slider('Selecione o horário', min_value=None, max_value=None, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=True, label_visibility="visible")
    else:
        st.write:(datetime.strptime(str(datetime.now()-timedelta(hours=3)),"%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M"))

utils.f_carrinho()

if st.button('Del item',disabled=state):
    if len(st.session_state.l_prod)==1:
        st.write (f"{st.session_state.l_prod.values}, removido do carrinho.")
        st.session_state.l_prod=[]
        state=True
        #placeholder.empty()
    else:    
        st.write (f"{st.session_state.l_prod[-1]}, removido do carrinho.")
        st.session_state.l_prod.pop()
        #placeholder.empty()  
        placeholder.text("Carrinho:")
        with placeholder.container():
            st.write('Carrinho:')
            for i in st.session_state.l_prod[0:-2]:
                st.write(i)

if st.button('Del carrinho',disabled=state):
        st.session_state.l_prod=[]
        st.write (f"Carrinho limpo.") 
        #placeholder.empty()


utils.r_np(st.session_state.df_lrecnp,st.session_state.l_prod,n=5)
utils.r_p(st.session_state.df_lrecnp,st.session_state.l_prod,st.session_state.user,n=5)