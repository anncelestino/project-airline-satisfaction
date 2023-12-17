import urllib.request

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import requests
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from pandas import json_normalize
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
from streamlit_player import st_player

### Reading CSVs
df = pd.read_csv('/Users/ann/Documents/coding-temple/05-week/project-airline-satisfaction/data/cleaned_train.csv')
df_test = pd.read_csv('/Users/ann/Documents/coding-temple/05-week/project-airline-satisfaction/data/cleaned_test.csv')
df1 = pd.read_csv('/Users/ann/Documents/coding-temple/05-week/project-airline-satisfaction/data/train.csv')
df2 = pd.read_csv('/Users/ann/Documents/coding-temple/05-week/project-airline-satisfaction/data/test.csv')



### Cleaning the data that had a hard time transfering to this app
df.drop(columns = ['Unnamed: 0'], inplace = True)
df_test.drop(columns = ['Unnamed: 0'], inplace = True)
df1.drop(columns = ['Unnamed: 0'], inplace = True)
df2.drop(columns = ['Unnamed: 0'], inplace = True)



# Set page title and icon
st.set_page_config(page_title = "✈︎ Analyzing Passenger Satisfaction Based on Airline Statistics", page_icon = '✈︎', layout = 'wide')



### Sidebar navigation
page = st.sidebar.selectbox("Select a Terminal", ["1. Home 🏡", "2. Data Overview 📚", "3. EDA 🔍", "4. Modeling ⚙️"])


### Adding music
st.write("🎵 Here's some inflight cabin music while you browse our app 💺 Buckle up and enjoy!")

audio_file1 = open('/Users/ann/Downloads/_The_Girl_from_Ipanema_Astrud_Gilberto_João_Gil.mp3', 'rb')
audio_file2 = open('/Users/ann/Downloads/Elton_John_Rocket_Man_Official_Music_Video_.mp3', 'rb')
audio_file3 = open('/Users/ann/Downloads/B_o_B_Airplanes_feat_Hayley_Williams_of_Paramo.mp3', 'rb')
audio_file4 = open('/Users/ann/Downloads/Leaving_On_a_Jet_Plane_Greatest_Hits_Version_.mp3', 'rb')
audio_bytes = audio_file1.read() + audio_file2.read() + audio_file3.read() + audio_file4.read()
st.audio(audio_bytes, format='audio/mp3')

st.write("---") 

### Build a homepage
if page == "1. Home 🏡":

    col1_spacer, col1, col2_spacer, col2, col3_spacer = st.columns([0.1, 2, 0.2, 1, 0.1])

    with col1:

        st.title("✈︎ Analyze Airline Passenger Experiences")
        st.subheader("Using ✨ The Passenger Airline Satisfaction Dataset Explorer App ✨")

    with col2:

        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()    
       
        lottie_plane = load_lottieurl("https://lottie.host/b236d22b-9047-4f9a-8fdf-fb3bb30c451d/xtT4IRQYw8.json")        
        st_lottie(lottie_plane, speed=1, height=250, key="initial")

    st.write("---") 

    col1_spacer, col1, col2, col3_spacer = st.columns([0.1, 1, 2, 0.1])
    
    with col1:

        st.subheader("✈︎ A Streamlit web app by [Ann Celestino]('https://github.com/anncelestino/project-airline-satisfaction')")

        # Load icon 2 
        lottie_plane2 = load_lottieurl("https://lottie.host/875a432b-86ec-4e05-b079-78ddcb4f452a/wxJ80xYcZP.json")        
        st_lottie(lottie_plane2, speed=1, height=150)

        st.write(">Fly with us to see the different data visualizations of passenger flights based on 'The Airline Passenger Satisfaction Dataset'!")
        
    with col2:

        st.image("https://www.travelandleisure.com/thmb/HbXDD_AhSlJXHm_iduNoUMPDxnU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/TAL-airport-gate-wait-layover-passenger-LAYOVERTIME0623--6c159c00aaf448de8baf97087490cad4.jpg")
        




### Build a Data Overview page
if page == "2. Data Overview 📚":

    st.title("🔢 Data Overview")

    col1_spacer, col1, col2_spacer, col2, col3_spacer = st.columns([0.1, 2, 0.2, 2, 0.1])
    
    with col1:
        st.subheader("About the Data")
        st.write(">This dataset consists of various columns on airline passenger flight information ranging from delays to satisfaction rates. The original dataset was obtained from Kaggle which you can find [here]('https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data'). The dataset can also be downloaded below!")
        # Download
        @st.cache_data
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(df1)
        st.download_button(
            label="Download dataset as CSV",
            data=csv,
            file_name='train.csv',
            mime='text/csv')
        
    with col2:
        st.image("https://www.nerdwallet.com/assets/blog/wp-content/uploads/2021/04/GettyImages-172672886.jpg")

    st.write("---")
    
    
    st.subheader("Quick Glance at the Airline Dataset 🧐")

    def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()  
    lottie_data_overview = load_lottieurl("https://lottie.host/ff8b9c06-101e-41e2-b2e2-5c0d0c0202d7/8ySYj8rXJN.json")        
    st_lottie(lottie_data_overview, speed=1, height=100, key="initial")

    container = st.container(border=True)
    container.write("Make your selection(s):")
    st.write("Here are your choices ↴")

    # Display dataset
    if container.checkbox("Data Frame"):
        st.dataframe(df1)

    # Column List
    if container.checkbox("Column List"):
        st.code(f"Columns: {df1.columns.tolist()}")

        if st.toggle('Further breakdown of columns'):
            num_cols = df1.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df1.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns: {obj_cols}")
    
    # Shape
    if container.checkbox("Shape"):
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")







# Build EDA page
if page == "3. EDA 🔍":

    col1_spacer, col1, col2, col3_spacer = st.columns([0.1, 2, 5, 0.1])

    with col1:
        st.title("🔍 EDA")

    with col2:
        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()  
        lottie_eda = load_lottieurl("https://lottie.host/1d1bb74c-d94f-40ef-81ef-34e183a92387/aPEu6GgfEQ.json")        
        st_lottie(lottie_eda, speed=1, height=200, key="initial")
    
    num_cols = df.select_dtypes(include = 'number').columns.tolist()
    obj_cols = df.select_dtypes(include = 'object').columns.tolist()

    container = st.container(border=True)

    

    eda_type = st.multiselect("What type of EDA are you interested in exploring?", 
                              ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])
    

    # COUNTPLOTS






    # HISTOGRAMS
    if "Histograms" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column", num_cols, index = None)

        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Beverage Category Hue on Histogram:"):
                st.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title, color = 'Beverage_category', barmode = 'overlay'))
            else:
                st.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title))
            if st.toggle("Beverage Name Hue on Histogram:"):
                st.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title, color = 'Beverage', barmode = 'overlay'))
            else:
                st.plotly_chart(px.histogram(df, x = h_selected_col, title = chart_title))


    # BOXPLOTS
    if "Box Plots" in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for your box plot:", num_cols, index = None)

        if b_selected_col:
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Beverage Category Hue on Box Plot"):
                st.plotly_chart(px.box(df, x = b_selected_col, y = 'Beverage_category', title = chart_title, color = 'Beverage_category'))
            else:
                st.plotly_chart(px.box(df, x = b_selected_col, title = chart_title))


    # SCATTERPLOTS
    if "Scatterplots" in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")

        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index = None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index = None)

        chart_title = f"Relationship of {selected_col_x} vs. {selected_col_y}"

        if selected_col_x and selected_col_y:
            col1, col2 = st.columns(2)
            col1, col2= st.columns([1, 3])
            col1.subheader("Selections 👇🏻")
            col2.subheader("Graphs 📊")
            
            with col1:
                toggle_1 = st.checkbox("Beverage Category Hue on Scatterplot")
                toggle_2 = st.checkbox("Beverage Name Hue on Scatterplot")
                if toggle_1:
                    with col2:
                        chart_title = f"Relationship of {selected_col_x} vs. {selected_col_y}"
                        st.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title, color = 'Beverage_category'))
                if toggle_2:
                    with col2:
                        st.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title, color = 'Beverage'))
            with col2:
                st.plotly_chart(px.scatter(df, x = selected_col_x, y = selected_col_y, title = chart_title))

# Build a Markdown Page
if page == "4. Modeling ⚙️":
    st.title(":gear: Modeling")
    st.markdown("**On this page, you can see how well different machine learning models** make predictions on the satisfaction rates!")

    # Set up X and y
    X = df.drop(columns = ['satisfaction'])
    y = df['satisfaction']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    model = RandomForestClassifier()
    
    k_value = st.slider("Select the number of neighbors (k)", 1, 29, 5, 2)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    if st.button("Let's see the performance!"):
        model.fit(X_train, y_train)

        # Display results
        st.subheader(f"{model} Evaluation")
        st.text(f"Training Accuracy: {round(model.score(X_train, y_train)*100, 2)}%")
        st.text(f"Testing Accuracy: {round(model.score(X_test, y_test)*100, 2)}%")

        # Confusion Matrix
        st.subheader("Confusion Matrix:")
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = 'Blues')
        st.pyplot()


st.write("---") 
st.write("🔧 Last Updated: Decemember 15, 2023") 

# Predictions Page
# if page == "Make Predictions!":
#     st.title(":rocket: Make Predictions on Iris Dataset")

#     # Create sliders for user to input data
#     st.subheader("Adjust the sliders to input data:")

#     s_l = st.slider("Sepal Length (cm)", 0.01, 10.0, 0.01, 0.01)
#     s_w = st.slider("Sepal Width (cm)", 0.01, 10.0, 0.01, 0.01)
#     p_l = st.slider("Petal Length (cm)", 0.01, 10.0, 0.01, 0.01)
#     p_w = st.slider("Petal Width (cm)", 0.01, 10.0, 0.01, 0.01)

#     # Your features must be in order that the model was trained on
#     user_input = pd.DataFrame({
#             'sepal_length': [s_l],
#             'sepal_width': [s_w],
#             'petal_length': [p_l],
#             'petal_width': [p_w]
#             })

#     # Check out "pickling" to learn how we can "save" a model
#     # and avoid the need to refit again!
#     features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#     X = df[features]
#     y = df['species']

#     # Model Selection
#     model_option = st.selectbox("Select a Model", ["KNN", "Logistic Regression", "Random Forest"], index = None)

#     if model_option:

#         # Instantiating & fitting selected model
#         if model_option == "KNN":
#             k_value = st.slider("Select the number of neighbors (k)", 1, 21, 5, 2)
#             model = KNeighborsClassifier(n_neighbors=k_value)
#         elif model_option == "Logistic Regression":
#             model = LogisticRegression()
#         elif model_option == "Random Forest":
#             model = RandomForestClassifier()
        
#         if st.button("Make a Prediction!"):
#             model.fit(X, y)
#             prediction = model.predict(user_input)
#             st.write(f"{model} predicts this iris flower is {prediction[0]} species!")
#             st.balloons()
