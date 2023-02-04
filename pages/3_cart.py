import streamlit as st
from datetime import datetime,  timedelta, time
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
    if 'clock' not in st.session_state:
        st.session_state.clock =(datetime.now()-timedelta(hours=3)).time()
    st.write('Simulação de periodo:')
    #clock=(datetime.now()-timedelta(hours=3))
    horario=st.checkbox('Horario atual', value=True, key=None, help='Marque para usar o horário local', on_change=None, args=None, kwargs=None, disabled=False)
    if horario:
       st.write("Relógio:",datetime.strptime(str(datetime.now()-timedelta(hours=3)),"%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M"))
       st.slider('Selecione o horário',min_value=time.min,max_value=time.max,value=(datetime.now()-timedelta(hours=3)).time(),format="HH:MM",disabled=True)
       st.session_state.clock =(datetime.now()-timedelta(hours=3)).time()
       hora=st.session_state.clock
       st.write('Horário adotado:',hora)
       #hora=st.slider('Selecione o horário', min_value=0, max_value=24,value=(datetime.now()-timedelta(hours=3)), step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=True)
    else:
       #hora=st.slider('Selecione o horário', min_value=0, max_value=24, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
        st.write("Relógio:",datetime.strptime(str(datetime.now()-timedelta(hours=3)),"%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M"))
        st.session_state.clock=st.slider('Selecione o horário',min_value=time.min,max_value=time.max,value=st.session_state.clock,format="HH:MM",disabled=False)
        hora=st.session_state.clock
        st.write('Horário adotado:',hora)
    
    

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

df_loja_rec=st.session_state.df_lrecnp
df_loja_rec['dth_hora'] = df_loja_rec['dth_agendamento'].apply(utils.extract_hour)
df_loja_rec=utils.time_filter(df_loja_rec)

utils.r_np(df_loja_rec,st.session_state.l_prod,n=5)
utils.r_p(df_loja_rec,st.session_state.l_prod,st.session_state.user,n=5)