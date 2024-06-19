import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import plotly.express as px
import streamlit as st
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

############# Importation des données
df = pd.read_csv('/Users/um/Desktop/Wild Code School/Protojam/dataOk.csv')

# Header de l'application
st.markdown("<h1 style='text-align: center; '>History of Olympic Games in few clicks</h1>", unsafe_allow_html=True)
st.image('https://img.freepik.com/photos-premium/paris-france-26-juillet-2024-anneaux-olympiques-quarelle_1002979-30782.jpg?w=740', use_column_width=True)

# Barres de sélection de données
season_options = ['Tous'] + sorted(df['Season'].unique())
season = st.sidebar.selectbox('Select a Season', season_options)

# Si une saison spécifique est sélectionnée, afficher les années correspondantes avec des JO
if season != 'Tous':
    years_with_olympics = df[df['Season'] == season]['Year'].unique()
else:
    years_with_olympics = df['Year'].unique()

year_options = ['Tous'] + sorted(years_with_olympics, reverse=True)
year = st.sidebar.selectbox('Select a Year', year_options)

# Filtrer les sports disponibles en fonction de la saison sélectionnée
if season != 'Tous':
    sports_available = df[df['Season'] == season]['Sport'].unique()
else:
    sports_available = df['Sport'].unique()

sport_options = ['Tous'] + sorted(sports_available)
sport = st.sidebar.selectbox('Select a Sport', sport_options)

# Filtrer les pays disponibles en fonction des autres sélections
if sport != 'Tous' and year != 'Tous':
    countries_available = df[(df['Sport'] == sport) & (df['Year'] == year)]['region'].unique()
else:
    countries_available = df['region'].unique()

country_options = ['Tous'] + sorted(countries_available)
country = st.sidebar.selectbox('Select a Country', country_options)

# Filtrer les genres disponibles en fonction des autres sélections
if country != 'Tous':
    genders_available = df[df['region'] == country]['Gender'].unique()
else:
    genders_available = df['Gender'].unique()

# Si "Tous" est sélectionné, inclure à la fois "Male" et "Female"
if 'Tous' in genders_available:
    gender_options = ['Tous', 'Male', 'Female']
else:
    gender_options = ['Tous'] + sorted(genders_available)

gender = st.sidebar.selectbox('Select a Gender', gender_options)

# Filtrer le DataFrame en fonction des sélections
conditions = (df['region'] == country) & (df['Sport'] == sport)

if season != 'Tous':
    conditions &= (df['Season'] == season)

if year != 'Tous':
    conditions &= (df['Year'] == year)

if gender != 'Tous':  
    conditions &= (df['Gender'] == gender)

filtered_df = df[conditions]

# Afficher le DataFrame filtré
if not filtered_df.empty:
    st.write(f'Olympics Games Distribution for {country} athletes- {gender} - {season} - {sport} in {year}')
    st.write(filtered_df.drop(columns=['NOC', 'Unnamed: 0', 'Country', 'Sport', 'Year', 'Season', 'Games']))
else:
    st.write("Aucune information disponible avec les filtres sélectionnés.")




#######  affichage du dataframe et des graphiques ##########
st.write('Podium')

if filtered_df.empty:
    st.write(f'Olympics Games Distribution for {country} athletes- {gender} - {season} - {sport} in {year}')
    st.write(filtered_df.drop(columns = ['NOC', 'Unnamed: 0', 'Country', 'Games']))
else:
    # Count the number of medals per sport
    st.write(f'Olympics Games Medal Distribution for {country} athletes- {gender} - {season} - {sport} in {year}')
    #medal_df['Medal'] = medal_df['Medal'].replace(0, 'Non médaillé')
    medal_count = filtered_df.groupby('Medal').size().reset_index(name='Medal Count')


    # Create the bar plot
    figure_1 = px.bar(
        medal_count,
        x = 'Medal',
        y= 'Medal Count',
        barmode = 'group',
        color='Medal',
        title=f'Olympics Games Medal Distribution for {country} - {gender} - {season} - {sport} in {year}'
    )

    # Display the plot in Streamlit
    st.write(medal_count)
    st.write(figure_1)

######### graphique d'evolution du nombre des athlètes et des medailles en au fil des ans ##############

medals_by_region = df.groupby('region').size().reset_index(name='Total Medals')
medals_by_region_sorted = medals_by_region.sort_values(by='Total Medals', ascending=False)
top_20_regions = medals_by_region_sorted.head(20)['region']
medals_filtered = df[df['region'].isin(top_20_regions)]
medals_grouped = medals_filtered.groupby(['Year', 'region']).size().reset_index(name='Medals')

fig = px.bar(medals_grouped, 
             x='region', 
             y='Medals', 
             animation_frame='Year', 
             title= "Evolution du nombre d'athlètes par pays au fil des olympiades (Top 20 régions)",
             labels={"Medals": "Nombre d'athlètes", "region": "région"},
             color = 'Medals',
             height = 450)
st.plotly_chart(fig)

#########

medals_only = df[df['Medal'] != 'non-medalist']
medals_by_region = medals_only.groupby('region').size().reset_index(name='Total Medals')
medals_by_region_sorted = medals_by_region.sort_values(by='Total Medals', ascending=False)
top_20_regions = medals_by_region_sorted.head(20)['region']
medals_filtered = medals_only[medals_only['region'].isin(top_20_regions)]
medals_grouped = medals_filtered.groupby(['Year', 'region']).size().reset_index(name='Medals')

fig_2 = px.bar(medals_grouped, 
             x='region', 
             y='Medals', 
             animation_frame='Year', 
             title='Evolution du nombre de medails par pays au fil des ans (Top 20 régions)',
             labels={'Medals': 'Nombre de médailles', 'region': 'région'},
             color = 'Medals',
             height = 600) 
fig_2.update_yaxes(range=[0,450])

st.plotly_chart(fig_2)


################### Essai de prediction #############

