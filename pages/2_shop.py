import streamlit as st
import utils

# import pandas as pd
# import teste_strealit_main
# from pathlib import Path
# from streamlit.source_util import (
#     page_icon_and_name, 
#     calc_md5, 
#     get_pages,
#     _on_pages_changed
# )
# from streamlit.components.v1 import html



# # create navigation
# def nav_page(page_name, timeout_secs=3):
#     #credits: https://stackoverflow.com/questions/73755240/streamlit-pages-on-button-press-not-on-sidebar
#     nav_script = """
#         <script type="text/javascript">
#             function attempt_nav_page(page_name, start_time, timeout_secs) {
#                 var links = window.parent.document.getElementsByTagName("a");
#                 for (var i = 0; i < links.length; i++) {
#                     if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
#                         links[i].click();
#                         return;
#                     }
#                 }
#                 var elasped = new Date() - start_time;
#                 if (elasped < timeout_secs * 1000) {
#                     setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
#                 } else {
#                     alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
#                 }
#             }
#             window.addEventListener("load", function() {
#                 attempt_nav_page("%s", new Date(), %d);
#             });
#         </script>
#     """ % (page_name, timeout_secs)
#     html(nav_script)


# def delete_page(main_script_path_str, page_name):

#     current_pages = get_pages(main_script_path_str)

#     for key, value in current_pages.items():
#         if value['page_name'] == page_name:
#             del current_pages[key]
#             break
#         else:
#             pass
#     _on_pages_changed.send()

# def add_page(main_script_path_str, page_name):
    
#     pages = get_pages(main_script_path_str)
#     main_script_path = Path(main_script_path_str)
#     pages_dir = main_script_path.parent / "pages"
#     script_path = [f for f in pages_dir.glob("*.py") if f.name.find(page_name) != -1][0]
#     script_path_str = str(script_path.resolve())
#     pi, pn = page_icon_and_name(script_path)
#     psh = calc_md5(script_path_str)
#     pages[psh] = {
#         "page_script_hash": psh,
#         "page_name": pn,
#         "icon": pi,
#         "script_path": script_path_str,
#     }
#     _on_pages_changed.send()

#store=st.session_state.store
#user=st.session_state.user
#df=st.session_state.df
#st.write(store)
#st.write(user)
#st.write(df)




    ##Seleção dos campos referente ao produto:
st.write('Selecione o produto para o carrinho:')
df_loja=st.session_state.df[st.session_state.df['loja_compra']==st.session_state.store]
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

if st.button('Carrinho'):
        utils.add_page('teste_strealit_main.py', 'cart')
        utils.nav_page('cart')