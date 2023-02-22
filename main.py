#We import libraries to use them in this file
import streamlit as st
import pandas as pd 
from streamlit_option_menu import option_menu
from multiprocessing import Value
from pages.main import tabs as tf
from PIL import Image

st.set_page_config(
   page_title="Vocado",
   page_icon="🥑",  
   layout="wide", 
   menu_items = {
         'Get Help': 'https://github.com/HenryLABFinalGrupo02/trabajofinal',
         'Report a bug': "https://github.com/HenryLABFinalGrupo02/trabajofinal",
         'About': "# This is a header. This is an *VOCADO* cool app!"})

with open('style.css') as f:
   st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


 ################
###### MENU ######
 ################

   with st.sidebar:
      st.image(Image.open('./image/logo_vocado.png'))
   
      selected = option_menu(None, ["Home", "My Business", "Competition", "Opportunities", "Add business"], 
      icons=['house', 'building', 'globe', 'star', 'plus'], 
      menu_icon="cast", default_index=0, orientation="vertical",
      styles={
           "container": {"padding": "0!important", 
                        "background-color": "#E4FFED"},
           "icon": {"color": "#F4C01E",
                     "font-size": "25px"}, 
           "nav-link": {"font-size": "25px", 
                        "margin":"0px", 
                        "--hover-color": "#109138", 
                        "font-family":"Sans-serif", 
                        "background-color": "#E4FFED"},
           "nav-link-selected": {"background-color": "#109138", 
                                 "font-style":"Sans-serif", 
                                 "font-weight": "bold",
                                 "color":"#FFFFFF"},
       })
   
#####################
## IMPORT FUNCTIONS ##
#####################
   
## HOME 
   if selected == "Home":
      st.title('Welcome to Vocado Admin Center')
      tf.metricas()
      
   ## My Business
   if selected == "My Business":
      st.title('Business Admin Center')
      tf.select_business()


   ## My Competition
   if selected == "Competition":
      st.title('Competition')
      tf.timeseries()

   ## My Opportunities
   if selected == "Opportunities":
      st.title('Opportunities Exploration')
      tf.machine_learning()

   ## Add Business
   if selected == "Add business":
      tf.addbusiness()