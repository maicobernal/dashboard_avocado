## Libraries importing
import streamlit as st
import pandas as pd
import pickle
import xgboost
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'
from sqlalchemy import create_engine
from transformers import AlbertForSequenceClassification, pipeline, AlbertTokenizer
from keybert import KeyBERT
import pydeck as pdk
import os
import json

## Function to connect to SupaBase Postgres DB
def connect_to_postgres():
    '''
    This function connects to the SupaBase DB
    using a json file with the credentials as a first try.
    If the file doesn't exist, then it will look for it on the enviroment variables.
    '''
    try:
        with open('credentials_supabase.json', 'r') as file:
            data = json.load(file)
            supabase_url = data['url']
            supabase_key = data['key']
            supabase_username = data['username']

        supabase_port = 5432
        supabase_db = "postgres"

        db_url = f"postgresql://{supabase_username}:{supabase_key}@{supabase_url}:{supabase_port}/{supabase_db}"

        engine_supabase = create_engine(db_url)
        print('Connected to SupaBase DB with JSON credentials')
        return engine_supabase
    except:
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_KEY')
        supabase_username = os.environ.get('SUPABASE_USERNAME')

        supabase_port = 5432
        supabase_db = "postgres"

        db_url = f"postgresql://{supabase_username}:{supabase_key}@{supabase_url}:{supabase_port}/{supabase_db}"

        engine_supabase = create_engine(db_url)
        print('Connected to SupaBase DB with environment variables')
        return engine_supabase

    

## Function to capitalize each word in a string
def capitalize_each_word(original_str):
    '''
    This function capitalizes each word in a string
    
    Parameters
    ----------
    original_str: str
        String to be capitalized
        
    Returns
    -------
    result: str
        String with each word capitalized'''

    result = ""
    # Split the string and get all words in a list
    list_of_words = original_str.split()
    # Iterate over all elements in list
    for elem in list_of_words:
        # capitalize first letter of each word and add to a string
        if len(result) > 0:
            result = result + " " + elem.strip().capitalize()
        else:
            result = elem.capitalize()
    # If result is still empty then return original string else returned capitalized.
    if not result:
        return original_str
    else:
        return result

## Function to get all data from previously setted business ids
## This business are the 'already owned' by the user
def update_my_businesses(ids_list:list):
    '''
    This function gets all data from the DB for the business ids that the user already owns
    
    Parameters
    ----------
    ids_list: list
        List of business ids that the user already owns

    Returns
    -------
    business: pd.DataFrame
        Dataframe with all business data
    checkin: pd.DataFrame
        Dataframe with all checkin data
    review: pd.DataFrame
        Dataframe with all review data
    sentiment: pd.DataFrame
        Dataframe with all sentiment data
    influencer_score: pd.DataFrame
        Dataframe with all influencer score data

    '''
    ## Getting data from DB
    engine = connect_to_postgres()
    tup = tuple(ids_list)

    ## If the user has more than one business, then we need to use the 'in' clause in the SQL query
    if len(tup) > 1:
        business = pd.read_sql("""SELECT * FROM business WHERE business_id in {}""".format(tup), engine)
        bus_names = business.name.to_list()
        for x in bus_names:
            users_business.add(x)
        checkin = pd.read_sql("""SELECT * FROM checkin WHERE business_id in {}""".format(tup), engine)
        review = pd.read_sql("""SELECT * FROM review WHERE business_id in {}""".format(tup), engine, parse_dates=['date'])
        sentiment = pd.read_sql("""SELECT * FROM sentiment WHERE business_id in {}""".format(tup), engine)
        influencer_score = pd.read_sql("""SELECT * FROM business_target WHERE business_id in {}""".format(tup), engine)
    
    ## If the user has only one business, then we need to use the '=' clause in the SQL query
    else:
        business = pd.read_sql("""SELECT * FROM business WHERE business_id = '{}'""".format(tup[0]), engine)
        users_business.add(business.name[0])
        checkin = pd.read_sql("""SELECT * FROM checkin WHERE business_id = '{}'""".format(tup[0]), engine)
        review = pd.read_sql("""SELECT * FROM review WHERE business_id = '{}'""".format(tup[0]), engine, parse_dates=['date'])
        sentiment = pd.read_sql("""SELECT * FROM sentiment WHERE business_id = '{}'""".format(tup[0]), engine)
        influencer_score = pd.read_sql("""SELECT * FROM business_target WHERE business_id = '{}'""".format(tup[0]), engine)
    

    return business, checkin, review, sentiment, influencer_score


## Set of business names that the user set in the tab 'Add Business' as 'owned' 
users_business = set()

## Setting the business ids that the user already owns
bus_ids = [1,7,9]

##### GETTING DATA FROM DB #####
business, checkin, review, sentiment, influencer_score = update_my_businesses(bus_ids)

############################################ HOME TAB ##################################################
# This is the first tab from streamlit app and gets the data from the DB and shows the metrics

def metricas(): 
    '''
    This function gets all data from the DB for the business ids that the user already owns
    It shows each business metrics and the total metrics for all businesses in streamlit'''

    ## Engine for DB connection
    engine = connect_to_postgres()

    ## Initial calculations for metrics
    review_stars = business['stars'].mean()
    sentiment['positive_score'] = sentiment['pos_reviews'] / ( sentiment['neg_reviews'] + sentiment['pos_reviews'])
    positive_sentiment = sentiment['positive_score'].mean()
    review_total = review.shape[0]
    number_visits = checkin['total'].sum()
    influencer_score['influencer_score'] = 1 - (1 / (1 + influencer_score['influencer_score']))
    inf_score = influencer_score['influencer_score'].mean()

    ## Oportunities
    ## Gives you the top categories and locations of business according to top 50 sucessfull businesses by success score

    ## Getting the top 50 sucessfull businesses from DB by sucess score
    df = pd.read_sql_query('select business_id from business_target order by sucesss_score desc limit 50;', engine)
    sucessfull_business = tuple(df['business_id'].unique().tolist())

    ## Getting rest of the features for this top 50 sucessfull businesses
    sucess_query = '''SELECT DISTINCT(postal_code), areas,
    restaurants, food, shopping, homeservices, beautyandspas, 
    nightlife, healthandmedical, localservices, bars, automotive 
    FROM business where business_id IN {};'''

    df_sucess_zip = pd.read_sql(sucess_query.format(sucessfull_business), engine)

    ## Transforming the areas column to the name of the area to visualization
    df_sucess_zip['areas_name'] = df_sucess_zip['areas'].map({
        0: 'Philadelphia',
        1: 'Reno',
        2: 'Indianapolis',
        3: 'Tucson',
        4: 'New Orleans',
        5: 'St. Louis',
        6: 'Tampa',
        7: 'Boise',
        9: 'Nashville',
        10: 'Santa Barbara'
    })

    areas = df_sucess_zip['areas_name'].value_counts().head(3).index.tolist()
    areas = ', '.join(areas)

    ## Getting the top 3 business lines
    subset = df_sucess_zip[['restaurants', 'food', 'shopping', 'homeservices', 'beautyandspas', 'nightlife', 'healthandmedical', 'localservices', 'bars', 'automotive']]
    business_lines = pd.get_dummies(subset).idxmax(1).value_counts().head(3).index.tolist()
    business_lines = ', '.join(business_lines)
    business_lines = capitalize_each_word(business_lines)
    
    ## Visualizing the metrics
    st.markdown("### Oportunities")
    oportunity = st.columns(2)
    oportunity[0].metric('Hot Business Lines', business_lines, delta=None, delta_color="normal")
    oportunity[1].metric('Hot Locations', areas, delta=None, delta_color="normal")

    ## Account Summary
    st.markdown("### Account Summary")
    metrics = st.columns(6)
    metrics[0].metric('Review Total', review_total, delta=None, delta_color="normal")
    metrics[1].metric('Review stars', round(review_stars, 2), delta=None, delta_color="normal")
    metrics[2].metric('Positive sentiment', f'{round(positive_sentiment, 2)*100}%', delta=None, delta_color="normal")
    metrics[3].metric('Influencer Score', f'{round(inf_score, 2)*100}%', delta=None, delta_color="normal")
    metrics[4].metric('Top Hour', '18:00', delta=None, delta_color="normal")
    metrics[5].metric('Visitors', number_visits)



############################################ BUSINESS TAB ##################################################
## This tab shows the business that the user owns and the metrics of each business
## Also the user can do sentiment analysis of reviews in real time

def query_info(filtro):
    '''
    This function visualizes metrics for the specific business id that the user selected
    in the streamlit app

    In the second part it does the sentiment analysis of the reviews in real time

    It is called in the function 'select_business'
    Where the user in the streamlit app selects the business which he wants to see the metrics

    Parameters
    ----------
    filtro : pd.DataFrame
        Dataframe with the business id that the user owns
    '''

    ## Preprocessing the data for visualization 
    ids = filtro['business_id'].to_list()

    review1 = review.loc[review['business_id'].isin(ids)]
    review_stars = filtro['stars'].mean()
    sentiment1 = sentiment.loc[sentiment['business_id'].isin(ids)]
    checkin1 = checkin.loc[checkin['business_id'].isin(ids)]
    influencer_score['influencer_score'] = 1 - (1 / (1 + influencer_score['influencer_score']))
    inf_score_1 = influencer_score.loc[influencer_score['business_id'].isin(ids)]
    sentiment1['positive_score'] = sentiment1['pos_reviews'] / ( sentiment1['neg_reviews'] + sentiment1['pos_reviews'])
    Positive_sentiment = sentiment1['positive_score'].mean()
    review_total = review1.shape[0]
    number_visits = checkin1['total'].sum()   
    inf_score_1 = inf_score_1['influencer_score'].mean()

    ## Visualizing the metrics
    st.markdown("### Account Summary")

    metrics = st.columns(6)
    metrics[0].metric('Review Total',review_total, delta=None, delta_color="normal")
    metrics[1].metric('Review stars', round(review_stars, 2), delta=None, delta_color="normal")
    metrics[2].metric('Positive sentiment', f'{round(Positive_sentiment, 2)*100}%', delta=None, delta_color="normal")
    if len(checkin1) > 1:
        metrics[4].metric('Top Hour', f'{round(checkin1.avg_hour.mean())}:00', delta=None, delta_color="normal")
    elif len(checkin1) == 1:
        metrics[4].metric('Top Hour', f'{round(checkin1.avg_hour.iloc[0])}:00', delta=None, delta_color="normal")
    metrics[3].metric('Influencer Score', f'{round(inf_score_1, 2)*100}%', delta=None, delta_color="normal")
    metrics[5].metric('Number_visits', number_visits)

    ## Visualizing the reviews and sentiment analysis
    st.title('Sentiment Analysis for Last Reviews')

    ## Getting the number of reviews that the user wants to analize
    number_to_get = st.slider('Number of reviews to get', 1, 50, 10)

    ## Doing the sentiment analysis
    name = st.button('Analize reviews')
    if name:
        reviews = review1[['text', 'date']].sort_values(by='date', ascending=False).head(number_to_get)
        
        ## Loading the model
        model = AlbertForSequenceClassification.from_pretrained('./model/textclass/')
        tokenizer = AlbertTokenizer.from_pretrained('./model/textclass/')

        ## Doing the sentiment analysis
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) #, device=0) #for GPU support
        
        ## Getting the keywords for each subset of reviews (positive and negative)
        kw_model = KeyBERT()
        positive = 0
        negative = 0
        pos_keywords = []
        neg_keywords = []
        reviews['sentiment'] = ''
        reviews['keywords'] = ''
        for index, row in reviews.iterrows():
            if classifier(row['text'], truncation = True)[0]['label'] == 'LABEL_1':
                positive += 1
                keywords = kw_model.extract_keywords(row['text'], keyphrase_ngram_range=(1, 1), stop_words='english')
                pos_keywords += keywords
                reviews.sentiment[index] = 'positive'
                reviews.keywords[index] = keywords
            elif classifier(row['text'], truncation = True)[0]['label'] == 'LABEL_0':
                negative += 1
                keywords = kw_model.extract_keywords(row['text'], keyphrase_ngram_range=(1, 1), stop_words='english')
                neg_keywords += keywords
                reviews.sentiment[index] = 'negative'
                reviews.keywords[index] = keywords
        
        try:
            neg_key, neg_score = zip(*neg_keywords)
        except:
            neg_key = []
            neg_score = []
        try:
            pos_key, pos_score = zip(*pos_keywords)
        except:
            pos_key = []
            pos_score = []

        ## Visualizing the results
        df_neg = pd.DataFrame({'key':neg_key, 'score':neg_score}).groupby('key').mean().sort_values('score', ascending=False)
        df_pos = pd.DataFrame({'key':pos_key, 'score':pos_score}).groupby('key').mean().sort_values('score', ascending=False)
            
        st.markdown("### Reviews Summary")
        metrics = st.columns(2)

        metrics[0].markdown("### Positive Reviews")
        metrics[0].metric('Total Positive', positive, delta=None, delta_color="normal")
        metrics[0].text("Top 5 Keywords")
        metrics[0].text(df_pos.head(5).index.tolist())

        metrics[1].markdown("### Negative Reviews")
        metrics[1].metric('Total Negative', negative, delta=None, delta_color="normal")
        metrics[1].text("Top 5 Keywords")
        metrics[1].text(df_neg.head(5).index.tolist())

        REVIEW_TEMPLATE_MD = """{} - {}
                                    > {}"""

        with st.expander("💬 Show Reviews"):

        # Show comments
            st.write("**Reviews:**")
            for index, entry in enumerate(reviews.itertuples()):
                st.markdown(REVIEW_TEMPLATE_MD.format(entry.date, entry.sentiment, entry.text))


def select_business():
    '''
    This function allows the user to select a business from the list of businesses
    that he owns
    '''
    ## Selectbox to choose the business
    option = st.selectbox(
            'My businesses',
            users_business)

    ## Querying the business info and showing it
    if option in option:
        filtro = business[business['name'] == option]
        query_info(filtro)





##################################### MODEL TAB ##################################################
## This tabs makes predictions in real time for business oportunities based on
#  the business features and predictions of the XGBOOST trained model

def machine_learning():
    st.markdown("Discover which business lines, location and services get you the better chances at being successful (Based on popularity)")

    st.header("Business Features")
    col1, col2, col3 = st.columns(3)

    ## Menu to select the business features which will be used to make the prediction

    ## Geographic area
    with col1:
        area = st.selectbox('Select geographical area', [
        'Philadelphia', 
        'Indianapolis', 
        'St. Louis', 
        'Nashville', 
        'New Orleans', 
        'Orlando',
        'Tucson', 
        'Santa Barbara', 
        'Reno', 
        'Boise'])

    ## Business type
        type_of_business = st.selectbox('Select type of business', [
            'Restaurant', 
            'Food', 
            'Nightlife', 
            'Shopping', 
            'Beauty & Spas', 
            'Bars',
            'Automotive', 
            'Health & Medical', 
            'Home Services', 
            'Local Services', 
            'Other'])

    ## Business line
    with col2:
        # st.text("Price range")
        price_range = st.slider('Price range', min_value = 0, max_value = 4, value = 1)

        # st.text("Noise level")
        noise_level = st.slider('Noise level', min_value = 0, max_value = 4, value = 1)

    ## Business services
    with col3: 
        # st.text("Open times")
        meal_diversity = st.slider('Meal diversity (if restaurant)', min_value = 0, max_value = 6, value = 1,
        help = "Meal diversity, 1 being only breakfast or dinner, 6 being all meals")

        open_hours = st.slider('Open Hours', min_value = 0.0, max_value = 24.0, value = 1.0, help= "Total open hours per day")
        
        weekends = st.checkbox(
        "Open on weekends",
        help="Weekends mean friday, saturday and sundays")

    st.header("Additional Features")
    col1, col2, col3 = st.columns(3)

    ## Additional features
    with col1:
        ambience = st.checkbox(
        "Good ambience",
        help="Comfortable, clean, peaceful, etc.")

        good_for_groups = st.checkbox(
        "Good for groups",
        help="Offers space for groups allocation")

        good_for_kids = st.checkbox(
        "Good for kids",
        help="Offers space for kids entertainment")
        
        has_tv = st.checkbox(
        "TV",
        help="Has TV")

        outdoor_seating = st.checkbox(
        "Outdoor seating",
        help="Outdoor seating")

    ## Additional features
    with col2:
        alcohol = st.checkbox(
        "Alcohol",
        help="Alcohol")

        delivery = st.checkbox(
        "Delivery",
        help="Delivery")

        garage = st.checkbox(
        "Garage",
        help="Garage")

        bike_parking = st.checkbox(
        "Bike parking",
        help="Offers parking locations for bikes")

        credit_cards = st.checkbox(
        "Credit cards",
        help="Accept credit cards")

    with col3:
        caters = st.checkbox(
        "Caters",
        help="Provides food service at a remote site")

        elegancy = st.checkbox(
        "Elegant",
        help="Provide elegant or formal ambience")

        by_appointment_only = st.checkbox(
        "Appointment",
        help="By appointment only")

        wifi = st.checkbox(
        "Wifi",
        help="Has Wifi")  

        reservations = st.checkbox(
        "Accept reservations",
        help="Accept reservations prior to attendance")  

    ## Transform the data to the same format as the training data
    if st.button('Predict Business Success'):

        areas_name = [
        'Philadelphia', 
        'Indianapolis', 
        'St. Louis', 
        'Nashville', 
        'New Orleans', 
        'Tampa',
        'Tucson', 
        'Santa Barbara', 
        'Reno', 
        'Boise']


        df_1 = pd.DataFrame({
            'ambience': [ambience],
            'garage': [garage],
            'credit_cards': [credit_cards],
            'bike_parking': [bike_parking],
            'wifi': [wifi],
            'delivery': [delivery],
            'good_for_kids': [good_for_kids],
            'outdoor_seating': [outdoor_seating],
            'reservations': [reservations],
            'has_tv': [has_tv],
            'good_for_groups': [good_for_groups],
            'alcohol': [alcohol],
            'by_appointment_only': [by_appointment_only],
            'caters': [caters],
            'elegancy': [elegancy],
            'noise_level': [noise_level],
            'meal_diversity': [meal_diversity]
        })

        df_1 = df_1.astype(int)

        df_2 = pd.DataFrame({   'Restaurants': [0.0],
                                                'Food': [0.0],
                                                'Shopping': [0.0],
                                                'Home Services': [0.0],
                                                'Beauty & Spas': [0.0],
                                                'Nightlife': [0.0],
                                                'Health & Medical': [0.0],
                                                'Local Services': [0.0],
                                                'Bars': [0.0],
                                                'Automotive': [0.0]})

        business_map = {
            'Restaurant': 'Restaurants',
            'Food': 'Food',
            'Shopping': 'Shopping',
            'Home Services': 'Home Services',
            'Beauty & Spas': 'Beauty & Spas',
            'Nightlife': 'Nightlife',
            'Health & Medical': 'Health & Medical',
            'Local Services': 'Local Services',
            'Bars': 'Bars',
            'Automotive': 'Automotive'
        }

        column_name = business_map.get(type_of_business)
        if column_name:
            df_2[column_name] = 1.0

        df_3 = pd.DataFrame({'weekends': [int(weekends)],
                            'open_hours': [float(open_hours)]})


        df_4 = pd.DataFrame({   'areas_0': [0.0],
                                    'areas_1': [0.0],
                                    'areas_2': [0.0],
                                    'areas_3': [0.0],
                                    'areas_4': [0.0],
                                    'areas_5': [0.0],
                                    'areas_6': [0.0],
                                    'areas_8': [0.0],
                                    'areas_9': [0.0],
                                    'areas_10': [0.0]})

        area_map = {
            'Philadelphia': 'areas_0',
            'Reno': 'areas_1',
            'Indianapolis': 'areas_2',
            'Tucson': 'areas_3',
            'New Orleans': 'areas_4',
            'St. Louis': 'areas_5',
            'Tampa': 'areas_6',
            'Boise': 'areas_7',
            'Nashville': 'areas_9',
            'Santa Barbara': 'areas_10'
        }

        column_name = area_map.get(area)
        if column_name:
            df_4[column_name] = 1.0

        

        df_5 = pd.DataFrame({'price_ranges_0': [0.0],
                            'price_ranges_1': [0.0],
                            'price_ranges_2': [0.0],
                            'price_ranges_3': [0.0],
                            'price_ranges_4': [0.0]})
                            
        price_range_map = {
            0: 'price_ranges_0',
            1: 'price_ranges_1',
            2: 'price_ranges_2',
            3: 'price_ranges_3',
            4: 'price_ranges_4'
        }

        column_name = price_range_map.get(price_range)
        if column_name:
            df_5[column_name] = 1.0

        df_predictions_final = pd.concat([df_1, df_2, df_3, df_4, df_5], axis = 1)

        ## Load model
        clf = pickle.load(open('./model/xgb_business.pkl', 'rb'))

        ## Get the names of the features as in the model
        df_predictions_final.columns = clf.get_booster().feature_names
        
        ## Predict
        prediction = clf.get_booster().predict(xgboost.DMatrix(df_predictions_final))
        
        ## Show the prediction probability
        st.success('Business probability of success: {prediction:.2f}'.format(prediction = prediction[0]))

        ## Show the prediction popular/not popular
        if prediction > 0.5:
            st.success('Business is popular')
        else:
            st.error('Business is not popular')




############################################## TIME SERIES TAB ############################################


def timeseries():
    '''
    This function is used to visualize the time series data
    Data is already preprocessed and saved in the data folder
    It only needs to be loaded and plotted
    It contains data for the top 20 brands in USA
    It is calculated by the total number of reviews/tips/checkins
    '''

    ## Load data
    train_df = pd.read_csv('./pages/main/data/train_ts.csv', index_col='month', parse_dates=True)
    forecast_df = pd.read_csv('./pages/main/data/forecast_ts.csv', index_col='month', parse_dates=True)
    type_model = pd.read_csv('./pages/main/data/model_for_each_ts.csv', index_col=0)

    ## Plot the time series data
    st.title('Time Series Visualization')
    st.markdown('Reviews/Tips/Checkins by Month for the Top Brands in USA')

    st.text("Select you favourite brand")
    top_brand_selected = st.multiselect('Select brand', train_df.columns.tolist(), train_df.columns.tolist()[0:3])

    st.plotly_chart(train_df[top_brand_selected].plot(title = 'Total Review/Tips/Checkins Counts on Yelp for Top Brands'))

    st.title('Forecasting Time Series')
    st.markdown('Reviews/Tips/Checkins by Month for the Top Brands in USA')

    st.text("Select you favourite brand")
    top_brand_selected_f = st.multiselect('Select brand for forecast', train_df.columns.tolist(), train_df.columns.tolist()[0:2])

    ## Plot the forecasts for the selected brands
    make_forecast = st.button('Make Forecasts')

    if make_forecast:
        for i in top_brand_selected_f:
            fig1 = px.line(train_df[i])
            fig1.update_layout(title='Actual')
            fig1.update_traces(line_color='purple', name='Actual')

            fig2 = px.line(forecast_df[i])
            fig2.update_layout(title='Forecast')
            fig2.update_traces(line_color='seagreen', name='Forecast')

            fig = go.Figure(data = fig1.data + fig2.data)
            fig.update_layout(title=i)
            st.plotly_chart(fig)
            texto = 'Type of model used for {}: {}'.format(i, type_model.loc[i, 'Best Model'])
            htmlcode = "<p style='text-align: center; color: red;'>{}</p>".format(texto)
            st.markdown(htmlcode, unsafe_allow_html=True)



############################################## Add business ############################################


def addbusiness():
    '''
    This function is used to add a new business to the streamlit app
    It is connected to the postgres database where the data is stored
    User can add a new business and it will be added to the local streamlit app
    It plots all the possible matches in a map
    '''
    
    global business, checkin, review, sentiment, influencer_score

    ## Engine to connect to the database
    engine = connect_to_postgres()

    ## Search for the business by name, then by CP and then by address
    st.markdown('#### Add your bussiness name')
    name = st.text_input('Add your business name 👇 and hit ENTER', '')
    if len(name) >2:
        query = f"SELECT name, address, postal_code, latitude_x as lat, longitude_x as lon, stars FROM business WHERE name = '{name}' ORDER BY postal_code ASC"
        df = pd.read_sql(query, con=engine)
        st.markdown('#### Map of the business matches')

        # create the Scattermapbox plot
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", hover_name="postal_code",
                                color = 'stars', size = 'stars'
                                , hover_data=['name','address', 'stars'], zoom = 3)

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True, theme = None)

        st.markdown('#### Postal Code')
        zipcode = st.selectbox('Select your postal code 👇', df['postal_code'].unique().tolist())

        st.markdown('#### Address')
        address = st.selectbox('Select your address 👇', df.loc[df['postal_code'] == zipcode,'address'].unique().tolist())

        add_my_bussiness = st.button('Add my business')

        if add_my_bussiness:
            new_business_id = pd.read_sql('SELECT business_id FROM business WHERE name = "{}" AND postal_code = "{}" AND address = "{}"'.format(name, zipcode, address), con=engine)['business_id'].values[0]
            
            bus_ids.append(new_business_id)
            
            users_business.add(capitalize_each_word(name))
            
            st.text('Your business id is: {}'.format(new_business_id))
            st.text('Business added to dashboard successfully')
            
            
            new_business, new_checkin, new_review, new_sentiment, new_influencer_score = update_my_businesses([new_business_id])
            business = pd.concat([business, new_business], axis=0)
            checkin = pd.concat([checkin, new_checkin], axis=0)
            review = pd.concat([review, new_review], axis=0)
            sentiment = pd.concat([sentiment, new_sentiment], axis=0)
            influencer_score = pd.concat([influencer_score, new_influencer_score], axis=0)

    else:
        st.text('Business not found, check the name and try again')