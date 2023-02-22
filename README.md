# **Dashboard**

- - -

This is a repository intended to get a fully functional dashboard from the final project at Henry, which can be explored [here](https://github.com/HenryLABFinalGrupo02/trabajofinal).

It only contains 10% of the data and it's uploaded to Supabase.com, a free PostgreSQL hosting. 

## **Context**

Yelp is a popular platform for reviews of all types of businesses, restaurants, hotels, services, among others.

We analyse the public dataset of reviews published by Yelp [here](https://www.yelp.com/dataset/documentation/main) which included data from 150K business with 1M users and 7M reviews. 

As the development was originally intended as a product for investors and local owners, we created a dashboard where users can select which business 'own' from the dataset an get statistics from them. 

We calculated a success score for business based on numbers of reviews, checkin and tips and the presence of influencer users. 

## **Dashboard structure**

#### Home
Here the users gets the features of the business with the highest sucess score (which business has more positive interactions with users). 

<img src="https://github.com/maicobernal/dashboard/blob/main/image/home.png"  height="300">

#### My Business
In this section users can select a particular business he/she owns and individual metrics for it. 
It can also analyze reviews (real time sentiment analysis) for up to last 50 reviews and get the keywords for positive and negative reviews. 
At the end the user can get the full review text if needed. 

<img src="https://github.com/maicobernal/dashboard/blob/main/image/mybusiness.png"  height="300">
<img src="https://github.com/maicobernal/dashboard/blob/main/image/sentiment.png"  height="300">

#### My Competition
In this section the user can get a time series analysis for the top 20 brands of the dataset to get insights about the number of reviews/checkin/tip for this companies. 
Time series analysis and forecasting was made and saved in CSV (model selection with MAPE metric). User can check forecasting visualizations for 2023. 

<img src="https://github.com/maicobernal/dashboard/blob/main/image/timeseries.png"  height="300">
<img src="https://github.com/maicobernal/dashboard/blob/main/image/forecast.png"  height="300">


#### Opportunities 
This section is intended for users who wants to get predictions for business oportunities for investment.
User can select business features and check if the business will be popular. Predictions are made with a trained XGBoost model.

<img src="https://github.com/maicobernal/dashboard/blob/main/image/oportunities.png"  height="300">
<img src="https://github.com/maicobernal/dashboard/blob/main/image/oportunities2.png"  height="300">


#### Add business
In this section the user can select another business preloaded in the database. Once selected, it will appear in "My Business" tab and can check reviews and metrics. 

<img src="https://github.com/maicobernal/dashboard/blob/main/image/add.png"  height="300">
<img src="https://github.com/maicobernal/dashboard/blob/main/image/add2.png"  height="300">

- - -
## **Data modeling**

<img src="https://github.com/maicobernal/dashboard/blob/main/image/modeling.png"  height="600">

- - -

