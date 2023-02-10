import streamlit as st
import utils


coverage_report, ranking_report, classification_report, rating_report= utils.master_m(st.session_state.df_lrecnp, filepath = 'pages/1ae58877-1bac-4e3e-9610-bc3c21b6d205')


st.title('Métricas de Acurácia')
tab1, tab2, tab3, tab4= st.tabs(["Classificação","Personalização e Ranqueamento","Feedback"])

with tab1:
    utils.plot_report(classification_report, figsize=(16,10))
with tab2:
    utils.plot_report(ranking_report, figsize=(16,10))
with tab3:
    utils.plot_report(rating_report, figsize=(16,10))

