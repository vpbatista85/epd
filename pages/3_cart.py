import streamlit as st
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