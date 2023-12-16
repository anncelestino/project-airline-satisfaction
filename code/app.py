import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Set page title and icon
st.set_page_config(page_title = "Welcome to The Airline Passenger Satisfaction Dataset Explorer", page_icon = '✈︎')

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home 🏡", "Data Overview 📚", "EDA 🔍", "Modeling"])

df = pd.read_csv('/Users/ann/Documents/coding-temple/05-week/project-airline-satisfaction/data/cleaned_train.csv')
df_test = pd.read_csv('/Users/ann/Documents/coding-temple/05-week/project-airline-satisfaction/data/cleaned_test.csv')
df1 = pd.read_csv('/Users/ann/Documents/coding-temple/05-week/project-airline-satisfaction/data/train.csv')
df2 = pd.read_csv('/Users/ann/Documents/coding-temple/05-week/project-airline-satisfaction/data/test.csv')

df.drop(columns = ['Unnamed: 0'], inplace = True)
df_test.drop(columns = ['Unnamed: 0'], inplace = True)
df1.drop(columns = ['Unnamed: 0'], inplace = True)
df2.drop(columns = ['Unnamed: 0'], inplace = True)

# Build a homepage
if page == "Home 🏡":
    st.title("🛩️ Airline Passenger Satisfaction Dataset Explorer App ✈︎")
    st.subheader("✨Welcome to our Passenger Airline Satisfaction Dataset Explorer App✨")
    st.write("> This app is designed to showcase different data visualizations of passenger flights based on The Airline Passenger Satisfaction Dataset!")
    st.image("https://www.nerdwallet.com/assets/blog/wp-content/uploads/2021/04/GettyImages-172672886.jpg")
    st.write("DISCLAIMER❗️: App is still a work in progress 🔧 Last Updated: Decemember 15, 2023")

# Build a Data Overview page
if page == "Data Overview 📚":
    st.title("🔢 Data Overview")
    st.subheader("About the Data")
    st.write("> This dataset consists of various columns on airline passenger flight information ranging from delays to satisfaction rates.")
    st.image("https://www.travelandleisure.com/thmb/HbXDD_AhSlJXHm_iduNoUMPDxnU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/TAL-airport-gate-wait-layover-passenger-LAYOVERTIME0623--6c159c00aaf448de8baf97087490cad4.jpg")
    st.write("> The original dataset was obtained from Kaggle. The dataset can also be downloaded below!")
    st.link_button("Kaggle Link", "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data", help = "Dataset Kaggle Page")
    
    # Download
    @st.cache_data
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df1)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='train.csv',
        mime='text/csv')

    
    st.subheader("Quick Glance at the Data 🧐")

    # Display dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df1)

    # Column List
    if st.checkbox("Column List"):
        st.code(f"Columns: {df1.columns.tolist()}")

        if st.toggle('Further breakdown of columns'):
            num_cols = df1.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df1.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns: {obj_cols}")
    if st.checkbox("Shape"):
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")

# Build EDA page
if page == "EDA 🔍":
    st.title("🔍 EDA")
    num_cols = df.select_dtypes(include = 'number').columns.tolist()
    obj_cols = df.select_dtypes(include = 'object').columns.tolist()

    eda_type = st.multiselect("What type of EDA are you interested in exploring?", 
                              ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])
    
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
if page == "Modeling":
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
