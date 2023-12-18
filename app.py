import urllib.request

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

import requests
import streamlit as st
import time

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
df = pd.read_csv('data/cleaned_train.csv')
df_test = pd.read_csv('data/cleaned_test.csv')
df1 = pd.read_csv('data/train.csv')
df2 = pd.read_csv('data/test.csv')

df3 = pd.read_csv('data/cleaned_test.csv')



### Cleaning the data that had a hard time transfering to this app
df.drop(columns = ['Unnamed: 0'], inplace = True)
df_test.drop(columns = ['Unnamed: 0'], inplace = True)
df1.drop(columns = ['Unnamed: 0'], inplace = True)
df2.drop(columns = ['Unnamed: 0'], inplace = True)





# Set page title and icon
st.set_page_config(page_title = "✈︎ Analyzing Passenger Satisfaction Based on Airline Statistics", page_icon = '✈︎', layout = 'wide')





### Sidebar navigation
page = st.sidebar.selectbox("Select a Destination", ["1. Home 🏡", "2. Data Overview 📚", "3. EDA 🔍", "4. Modeling ⚙️", "5. Make Predictions! 🔮"])




### Adding music
st.write("🎵 Here's some inflight cabin music while you browse our app 💺 Buckle up and enjoy!")

audio_file = open('audio/Frank_Sinatra_Come_Fly_With_Me_1958_.mp3', 'rb')
# audio_file1 = open('audio/_The_Girl_from_Ipanema_Astrud_Gilberto_João_Gil.mp3', 'rb')
# audio_file2 = open('audio/The_Byrds_Eight_Miles_High_Audio_.mp3', 'rb')
# audio_file3 = open('audio/Elton_John_Rocket_Man_Official_Music_Video_.mp3', 'rb')
# audio_file4 = open('audio/B_o_B_Airplanes_feat_Hayley_Williams_of_Paramo.mp3', 'rb')
# audio_file5 = open('audio/Leaving_On_a_Jet_Plane_Greatest_Hits_Version_.mp3', 'rb')
# audio_file6 = open('audio/Galantis_Mama_Look_At_Me_Now_Official_Audio_.mp3', 'rb')
audio_bytes = (audio_file.read()) 
#+ audio_file1.read() + audio_file2.read() + audio_file3.read() + audio_file4.read() + audio_file5.read() + audio_file6.read())
st.audio(audio_bytes, format='audio/mp3')

st.write("---") 





### Build a homepage
if page == "1. Home 🏡":

    col1_spacer, col1, col2_spacer, col2, col3_spacer = st.columns([0.1, 2, 0.2, 1, 0.1])

    with col1:

        st.title("**✈︎ :orange[Analyze Airline Passenger Experiences]**")
        st.subheader(":grey[Using] ✨ The Airline Passenger Satisfaction Dataset Explorer App ✨")

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

        st.subheader("✈︎ :rainbow[A Streamlit web app] :grey[by] [Ann Celestino](https://github.com/anncelestino/project-airline-satisfaction)")

        # Load icon 2 
        lottie_plane2 = load_lottieurl("https://lottie.host/875a432b-86ec-4e05-b079-78ddcb4f452a/wxJ80xYcZP.json")        
        st_lottie(lottie_plane2, speed=1.5, height=150)

        st.write(">Fly with us to your destination! This app will take you on a journey of data analytical discovery through 'The Airline Passenger Satisfaction Dataset'!")
    with col2:

        st.image("https://www.travelandleisure.com/thmb/HbXDD_AhSlJXHm_iduNoUMPDxnU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/TAL-airport-gate-wait-layover-passenger-LAYOVERTIME0623--6c159c00aaf448de8baf97087490cad4.jpg")
    st.divider()
    container = st.container(border=True)
    container.write("💡 **:red[TIP]**: Get started by selecting a Destination Page by clicking on the arrow on the top left corner. I recommend working your way down the list, starting from the Data Overview Page to the Make Predictions Page, to get the full experience. :orange[Enjoy]! 🎈")
    st.write("***This was originally created for my Week 5: Intro to ML - Classification project at Coding Temple for the purpose of creating an interactive and user-friendly app with the primary focus of analyzing and predicting customer satisfaction.***")
    container2 = st.container(border=True)




### Build a Data Overview page
if page == "2. Data Overview 📚":
    st.title("🔢 **Data Overview** ")

    col1_spacer, col1, col2_spacer, col2, col3_spacer = st.columns([0.1, 2, 0.2, 2, 0.1])
    
    with col1:
        st.subheader("**:blue[About the Data] ✈**")
        st.write(">This dataset consists of various columns on airline passenger flight information ranging from flight distance, seat comfort, inflight entertainment, and more. **The original dataset was obtained from Kaggle which you can find [here](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data). The dataset can also be downloaded below!**")
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
        st.image("https://imageio.forbes.com/specials-images/imageserve/650bde4119f477fad93349ea/Tampa-International-Airport-named-the-best-Large-airport-in-America-/960x0.jpg?height=473&width=711&fit=bounds")

    st.write("---")
    
    col1,col2 = st.columns([2,1])
    with col1:
        st.subheader("Take a Quick Glance at the Airline Dataset 🧐")
        st.write(">**Check out the :blue[dataset] we are working with down below!**")
    with col2:
        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json() 
        lottie_data_overview = load_lottieurl("https://lottie.host/ff8b9c06-101e-41e2-b2e2-5c0d0c0202d7/8ySYj8rXJN.json")        
        st_lottie(lottie_data_overview, speed=1, height=200, key="initial")

    container = st.container(border=True)
    container2 = st.container(border=True)
    container.write("**Make your :red[selection(s)]:**")

    container2.write("Here are your choices ↴")

    # Display dataset
    if container.checkbox("**:blue[Data Frame]**"):
        container2.dataframe(df1)
    # Column List
    if container.checkbox("**:green[Column List]**"):
        container2.code(f"Columns: {df1.columns.tolist()}")
        if container2.toggle('**:grey[Further breakdown of columns]**'):
            num_cols = df1.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df1.select_dtypes(include = 'object').columns.tolist()
            container2.code(f"Numerical Columns: {num_cols} \nObject Columns: {obj_cols}")
    # Shape
    if container.checkbox("**:orange[Shape]**"):
        container2.write(f"**:grey[There are] :orange[{df1.shape[0]}] rows :grey[and] :orange[{df1.shape[1]}] columns.**")
    

        
    





# Build EDA page
if page == "3. EDA 🔍":

    col1_spacer, col1, col2, col3_spacer = st.columns([0.1, 2, 5, 0.1])

    with col1:
        st.title("🔍 Exploratory Data Analysis (EDA) ✈")
        st.header("**Let your :rainbow[curiosity] take over!**")

    with col2:
        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()  
        lottie_eda = load_lottieurl("https://lottie.host/1d1bb74c-d94f-40ef-81ef-34e183a92387/aPEu6GgfEQ.json")        
        st_lottie(lottie_eda, speed=1, height=450, key="initial")
    
    st.write(">**In this page, you can :blue[explore] the data by inputing :green[features] into the select boxes, which automatically generates different :orange[data visualizations] based on your selected choices!**")

    st.divider()

    num_cols = df1.select_dtypes(include = 'number').columns.tolist()
    obj_cols = df1.select_dtypes(include = 'object').columns.tolist()
    all_cols = df1.select_dtypes(include = ['number','object']).columns.tolist()

    container = st.container(border=True)

    eda_type = container.multiselect("What type of EDA are you interested in exploring?", 
                              ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])
    


    # COUNTPLOTS
    if "Count Plots" in eda_type:
        st.subheader("Count Plots - Visualizing Relationships")
        st.write("---")
        
        col1, col2 = st.columns(2)
        col1, col2= st.columns([1, 3])
        col1.subheader("Selections 👇🏻")
        col2.subheader("Graphs 📊")

        with col1:
            container = st.container(border=True)
            c_selected_col = container.selectbox("Select a categorical column for your countplot:", obj_cols, index = None)
            if c_selected_col:
                c_selected_col2 = container.selectbox("Select another categorical column:", obj_cols, index = None)
                with col2:
                    container2 = st.container(border=True)
                    if c_selected_col2:
                        with col2:
                            chart_title = f"Count of {' '.join(c_selected_col.split('_')).title()} & {' '.join(c_selected_col2.split('_')).title()}"
                            fig = st.plotly_chart(px.bar(df1, x = c_selected_col, title = chart_title, color = c_selected_col2, barmode='group'))
                    else:
                        with col2:
                            chart_title = f"Count of {' '.join(c_selected_col.split('_')).title()}"
                            num_value = df1[f"{c_selected_col}"].value_counts()
                            fig = st.plotly_chart(px.bar(df1, x = c_selected_col, title = chart_title))
                


    # HISTOGRAMS
    if "Histograms" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        st.write("---")
        
        col1, col2 = st.columns(2)
        col1, col2= st.columns([1, 3])
        col1.subheader("Selections 👇🏻")
        col2.subheader("Graphs 📊")

        with col1:
            container = st.container(border=True)
            h_selected_col = container.selectbox("Select a numerical column for your histogram:", num_cols, index = None)
            if h_selected_col:
                h_selected_col2 = container.selectbox("Select a hue:", obj_cols, index = None)
                with col2:
                    container2 = st.container(border=True)
                    if h_selected_col2:
                        with col2:
                            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()} Based On {' '.join(h_selected_col2.split('_')).title()} "
                            fig = container2.plotly_chart(px.histogram(df1, x = h_selected_col, title = chart_title, barmode = 'overlay', color = h_selected_col2))
                    else:
                        with col2:
                            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
                            fig = container2.plotly_chart(px.histogram(df1, x = h_selected_col, title = chart_title))



    # BOXPLOTS
    if "Box Plots" in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        st.write("---")
        
        col1, col2 = st.columns(2)
        col1, col2= st.columns([1, 3])
        col1.subheader("Selections 👇🏻")
        col2.subheader("Graphs 📊")

        with col1:
            container = st.container(border=True)
            b_selected_col = container.selectbox("Select a numerical column for your box plot:", num_cols, index = None)
            if b_selected_col:
                b_selected_col2 = container.selectbox("Select a hue:", obj_cols, index = None)
                with col2:
                    container2 = st.container(border=True)
                    if b_selected_col2:
                        with col2:
                            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()} Based On {' '.join(b_selected_col2.split('_')).title()} "
                            fig = container2.plotly_chart(px.box(df1, x = b_selected_col, y = b_selected_col2, title = chart_title, color = b_selected_col2))
                    else:
                        with col2:
                            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
                            fig = container2.plotly_chart(px.box(df1, x = b_selected_col, title = chart_title))



    # SCATTERPLOTS
    if "Scatterplots" in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        st.write("---")

        col1, col2 = st.columns(2)
        col1, col2= st.columns([1, 3])
        col1.subheader("Selections 👇🏻")
        col2.subheader("Graphs 📊")

        with col1:
            container = st.container(border=True)
            selected_col_x = container.selectbox("Select x-axis variable:", num_cols, index = None)
            selected_col_y = container.selectbox("Select y-axis variable:", num_cols, index = None)
            if selected_col_x and selected_col_y:
                selected_col_hue = container.selectbox("Select a hue:", obj_cols, index = None)
                with col2:
                    container2 = st.container(border=True)
                    chart_title = f"Relationship of {selected_col_x} vs. {selected_col_y}"
                    if selected_col_hue:
                        with col2:
                            container2.plotly_chart(px.scatter(df1, x = selected_col_x, y = selected_col_y, title = chart_title, color = selected_col_hue, opacity = 0.5))
                    else:
                        with col2:
                            container2.plotly_chart(px.scatter(df1, x = selected_col_x, y = selected_col_y, title = chart_title, opacity = 0.5))





# Build Modeling Page
if page == "4. Modeling ⚙️":
    st.title("🤖 Modeling ✈")
    st.markdown(">**On this page, you can see how well different :blue[machine learning models] make :violet[predictions] on :orange[customer satisfaction rates]!**")

    def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()  
    lottie_m = load_lottieurl("https://lottie.host/5685aad0-58cd-4833-acde-7f2967214db7/bbFj1U1YO9.json")        
    st_lottie(lottie_m, speed=1, height=200, key="initial")

    # Set up X and y
    X = df.drop(columns = ['satisfaction'])
    y = df['satisfaction']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # model = RandomForestClassifier()

    container = st.container(border=True)
    model_option = container.selectbox("1) Select a Model", ["KNN", "Logistic Regression", "Random Forest"], index = None)

    # Instantiating & fitting selected model
    if model_option:
        press_button = st.button("Let's see the performance!")
        if model_option == "KNN":
            k_value = container.slider("Select the number of neighbors (k)", 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors= k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        if press_button:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            model.fit(X_train, y_train)

            # Display results
            container2 = st.container(border=True)
            container2.subheader(f"{model} Evaluation")
            container2.text(f"Training Accuracy: {round(model.score(X_train, y_train)*100, 2)}%")
            container2.text(f"Testing Accuracy: {round(model.score(X_test, y_test)*100, 2)}%")

            # Confusion Matrix
            st.subheader("Confusion Matrix:")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = 'Blues')
            st.pyplot()

    container2 = st.container(border=True)
    container2.write("***Compare to the Baseline Score***")
    container2.write(":red[Neutral or Dissatisfied]: 56.61%")
    container2.write(":orange[Satisfied]: 43.34%")

# Build Predictions Page
if page == "5. Make Predictions! 🔮":
    st.title("🔮 Predictions ✈")
    st.markdown(">**On this page, you can make :violet[predictions] whether or not the passenger will be :rainbow[satisified] or :red[neutral/dissatisfied] with their flight based on your inputs using the Model of your choice!**")
    def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()    
       
    lottie_pred = load_lottieurl("https://lottie.host/ffc45eaa-d601-4b32-bf2c-c003969e5fa0/lc5SkBrQYf.json")        
    st_lottie(lottie_pred, speed=1, height=250, key="initial")
    st.divider()

    container1 = st.container(border=True)
    container2 = st.container(border=True)
    container3 = st.container(border=True)
    container4 = st.container(border=True)

    container2.subheader("Who is the Passenger? Input your passenger's info on the form below ⬇")
    container2.divider()
    container2.subheader("Input an Number")
    age_num = container2.number_input("What\'s your age? Pick an age from 1 to 100", min_value = 1, max_value = 100, step = 1, value=None, placeholder="Type a number...")
    if age_num:
        container2.write(f'The age is {age_num}')
    else:
        container2.write(":red[Please enter the age]")

    container2.subheader("Select Your Categories")
    sex = container2.radio("What's your sex?",
    ["Male", "Female"], index=None,)
    if sex:
        container2.write(f'You selected: **{sex}**')
        if sex == "Male":
            sex = 1
        if sex == "Female":
            sex = 0
    cl = container2.radio("Pick a class", ["Business", "Eco Plus", "Eco"], index=None)
    if cl:
        container2.write(f'You are flying in **{cl}** class')
        if cl == "Business":
            c_b = 1
        if cl != "Business":
            c_b = 0
        if cl == "Eco Plus":
            c_ep = 1
        if cl != "Eco Plus":
            c_ep = 0
        if cl == "Eco":
            c_e = 1
        if cl != "Eco":
            c_e = 0
    t_t = container2.radio("Pick a travel type", ["Business", "Personal"], index=None)
    if t_t:
        container2.write(f'This is a **{t_t}** travel')
        if t_t == "Business":
            t_t = 1
        if t_t == "Personal":
            t_t = 0
    c_t = container2.radio("Are you loyal to one airline?", ["Yes", "No"], index=None)
    if c_t == "Yes":
        container2.write(f'You are a **loyal customer**')
        c_t = 1
    if c_t == "No":
        container2.write('You are a **disloyal customer**')
        c_t = 0

    # Create sliders for user to input data
    col1, col2 = st.columns([3,1])

    with col1:
        container3.subheader("Adjust the sliders to input your :red[Satisfaction Levels] (1-5) for each category:")
        inf = container3.slider("Inflight Wifi Service 🛜", 1, 5, 3, 1)
        d_a = container3.slider("Departure/Arrival Time Convenient 🛫", 1, 5, 3, 1)
        o_b = container3.slider("Online Booking 💻", 1, 5, 3, 1)
        g_l = container3.slider("Gate Location 🛄", 1, 5, 3, 1)
        f_d = container3.slider("Food and Drink 🥜", 1, 5, 3, 1)
        o_b = container3.slider("Online Boarding 👩🏻‍💻", 1, 5, 3, 1)
        s_c = container3.slider("Seat Comfort 💺", 1, 5, 3, 1)
        inf_e = container3.slider("Inflight Entertainment 📺", 1, 5, 3, 1)
        o_s = container3.slider("On-board Service 👨🏼‍✈️", 1, 5, 3, 1)
        l_r = container3.slider("Leg Room Service 🦵🏼", 1, 5, 3, 1)
        b_h = container3.slider("Baggage Handling 🧳", 1, 5, 3, 1)
        c_s = container3.slider("Check-in Service ✔️", 1, 5, 3, 1)
        inf_s = container3.slider("Inflight Service 👩🏻‍✈️", 1, 5, 3, 1)
        cl = container3.slider("Cleanliness 🫧", 1, 5, 3, 1)
        
    with col2:
        container4.subheader("Flight Distance and Delays")
        fl = container4.slider("Flight Distance in Miles ✈️", 10, 5000, 10, 10)
        m_d = container4.slider("Departure Delay in Minutes ⏰", 0, 1600, 0, 1)
        m_a = container4.slider("Arrival Delay in Minutes 🛬", 0, 1600, 0, 1)

    # Your features must be in order that the model was trained on
    user_input = pd.DataFrame({
                'id':[42],
                'gender':[sex],
                'customer_type':[c_t],
                'age':[age_num],
                'type_of_travel':[t_t],
                # 'class':[cl],
                'flight_distance':[fl],
                'inflight_wifi_service':[inf],
                'departure/arrival_time_convenient':[d_a],
                'ease_of_online_booking':[o_b],
                'gate_location':[g_l],
                'food_and_drink':[f_d],
                'online_boarding':[o_b],
                'seat_comfort':[s_c],
                'inflight_entertainment':[inf_e],
                'on-board_service':[o_s],
                'leg_room_service':[l_r],
                'baggage_handling':[b_h],
                'checkin_service':[c_s],
                'inflight_service':[inf_s],
                'cleanliness':[cl],
                'departure_delay_in_minutes':[m_d],
                'arrival_delay_in_minutes':[m_a],
                'class_Business':[c_b],
                'class_Eco':[c_e],
                'class_Eco Plus':[c_ep]})

    
    # Check out "pickling" to learn how we can "save" a model
    # and avoid the need to refit again!
    
    X = df.drop(columns=['satisfaction'])
    y = df['satisfaction']

    # Model Selection
    container5= st.container(border=True)
    container5.header("Time to Fit Our Model!")
    model_option = container5.selectbox("Select a Model", ["KNN", "Logistic Regression", "Random Forest"], index = None)
    
    if model_option:

        # Instantiating & fitting selected model
        if model_option == "KNN":
            k_value = container5.slider("Select the number of neighbors (k)", 1, 21, 5, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        
        # def make_prediction():
            # msg = st.toast('Gathering the data...')
            # time.sleep(1)
            # msg.toast('Analyzing...')
            # time.sleep(1)
            # msg.toast('Ready!', icon = "🔮")

        if container5.button("Make a Prediction!"):
            model.fit(X, y)
            prediction = model.predict(user_input)
        # make_prediction()
            with st.spinner('Wait for it...'):
                time.sleep(5)
                st.success('Done!')
            if prediction == "neutral or dissatisfied":
                container5.header(f"{model} predicts that your passenger will be :red[{prediction[0]}] with their airline flight.")
                st.snow()
            if prediction == "satisfied":
                container5.header(f"{model} predicts that your passenger will be :rainbow[{prediction[0]}] with their airline flight!")
                st.balloons()
            else:
                container5.header("The Passenger Form is :red[missing] some information. Please recheck your inputs.")

st.divider()
st.write("🔧 Last Updated: Decemember 18, 2023")