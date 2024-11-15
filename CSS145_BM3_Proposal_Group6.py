# -------------------------

# Library Imports

# Streamlit
import streamlit as st
import io

# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.utils import resample

# Importing Models
import joblib

# Images
from PIL import Image


# -------------------------


# Functions

# `key` parameter is used to update the plot when the page is refreshed

def pie_chart_CrmCdDesc(df, width, height, key, top_n=10):

    # Count the occurrences of each unique value in 'Crm Cd Desc'
    CrmCdDesc_counts = df['Crm Cd Desc'].value_counts()
    CrmCdDesc_counts.columns = ['Crm Cd Desc', 'count']
    
    top_counts = CrmCdDesc_counts.head(top_n)
    others_count = pd.Series({'Others': CrmCdDesc_counts.iloc[top_n:].sum()})
    
    data = pd.concat([top_counts, others_count]).reset_index()
    data.columns = ['Crm Cd Desc', 'count']

    # Create the pie chart
    pie_chart = px.pie(
        data_frame=data,
        names='Crm Cd Desc',
        values='count',
        title='Pie Chart of Crime Code Description',
        color_discrete_sequence=px.colors.sequential.Plasma
    )

    # Update the layout for better appearance
    pie_chart.update_layout(
        width=width,
        height=height,
        showlegend=True,
        legend_title="Crime Description"
    )

    # Display the pie chart in Streamlit
    st.plotly_chart(pie_chart, use_container_width=True, key=f"pie_chart_{key}")
    
    
    
def bar_graph(df, column, width, height, key):
    
    value_counts = df[column].value_counts().sort_values(ascending=False)
    
    bar_chart = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        color=value_counts.index,
        labels={  # Custom labels for axes
            "x": column,
            "y": "Number of Crimes"
        },
        title=f"Number of Crimes per {column}",
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    
    bar_chart.update_layout(
        width=width,
        height=height,
        xaxis_title=column,
        yaxis_title="Number of Crimes",
        showlegend=False
    )
    
    st.plotly_chart(bar_chart, use_container_width=True, key=f"bar_chart_{key}")
    
    
def clean_data(df):
    # Drop columns with excessive missing values
    cols_to_drop = ['Mocodes', 'Weapon Used Cd', 'Weapon Desc', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'Cross Street']
    df = df.drop(columns=cols_to_drop)

    # Drop unnecessary columns
    cols_to_drop = ['DR_NO', 'Date Rptd', 'Rpt Dist No', 'Status', 'Status Desc', 'Crm Cd 1', 'LOCATION', 'LAT', 'LON']
    df = df.drop(columns=cols_to_drop)

    # Replace -1 and -2 values in the 'Vict Age' column with 0
    df['Vict Age'] = df['Vict Age'].replace([-1, -2], 0)

    # Impute missing values in 'Vict Sex' and 'Vict Descent'
    df.fillna({'Vict Sex': '-'}, inplace=True)
    df.fillna({'Vict Descent': '-'}, inplace=True)

    # Drop rows with missing values in 'Premis Cd' and 'Premis Desc'
    df.dropna(subset=['Premis Cd'], inplace=True)
    df.dropna(subset=['Premis Desc'], inplace=True)

    # Drop rows with 803.0 and 805.0 in 'Premis Cd'
    df.drop(df[df['Premis Cd'].isin([803.0, 805.0])].index, inplace=True)

    return df

    
def plot_age_group_distribution(df):
    # Filter out rows where Victim Age is 0 or less (no direct victim)
    age_filtered_df = df[df['Vict Age'] > 0]

    # Define age bins and labels for age groups
    age_bins = [0, 17, 30, 45, 60, 120]
    age_labels = ['Under 18', '18-30', '31-45', '46-60', 'Above 60']

    # Create a new column 'Age Group' using pd.cut
    age_filtered_df['Age Group'] = pd.cut(
        age_filtered_df['Vict Age'], bins=age_bins, labels=age_labels, right=False
    )

    # Count the number of occurrences for each age group
    crime_by_age_group = age_filtered_df['Age Group'].value_counts().sort_index()

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Age Group': crime_by_age_group.index,
        'Number of Crimes': crime_by_age_group.values
    })

    # Create the bar plot using Plotly Express
    fig = px.bar(
        plot_df,
        x='Age Group',
        y='Number of Crimes',
        title='Crimes Committed by Victim Age Group',
        color='Age Group',
        color_discrete_sequence=px.colors.sequential.Magma
    )

    # Update the layout for better appearance
    fig.update_layout(
        xaxis_title='Age Group',
        yaxis_title='Number of Crimes',
        legend_title='Age Group'
    )

    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
def balance_dataset(df, target_column):
    """Balances a dataset by resampling minority classes.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.

    Returns:
        pd.DataFrame: The balanced DataFrame.
    """

    class_counts = df[target_column].value_counts()
    min_samples = class_counts.min()

    balanced_df_list = []
    for class_label in class_counts.index:
        class_df = df[df[target_column] == class_label]
        resampled_df = resample(
            class_df,
            replace=True,
            n_samples=min_samples,
            random_state=42
        )
        balanced_df_list.append(resampled_df)

    balanced_df = pd.concat(balanced_df_list)
    balanced_df = balanced_df.sample(frac=1, random_state=42)

    return balanced_df

def feature_importance_plot(feature_importance_df, width, height, key):
    # Generate a bar plot for feature importances
    feature_importance_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        orientation='h'  # Horizontal bar plot
    )

    # Adjust the height and width
    feature_importance_fig.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot_{key}")
    

# -------------------------

# Page configuration
st.set_page_config(
    page_title="Los Angeles Crime Classification", 
    page_icon="CSS145_BM3_Proposal3_Group6/assets/icon/handcuff.png",
    layout="wide",
    initial_sidebar_state="expanded")

# -------------------------

# Sidebar

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.title('Los Angeles Crime Classification')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of a training a forest classification model using the Los Angeles Crime Analysis dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/nathaniellybrand/los-angeles-crime-dataset-2020-present)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1b1sZBo6abbv3cw3z_CA6pVPB7Tn-aV0c?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/chocomint04/CSS145-PROPOSAL-3)")
    st.subheader("Members")
    st.markdown("1. Adrian Besario\n2. Keane Benito\n3. Joaquin Anton Labao\n4. Erin Brent Limpiada\n5. Rishon Simone Papa")

# -------------------------

# Data

# Load data
crime_df = pd.read_csv("data/Crime_Data_from_2020_to_Present.csv")


# -------------------------


# Importing models

model = joblib.load("assets/model/crime_type_predictor_model.joblib")


# -------------------------


# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown(""" 

    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to classify Los Angeles Crime Type from the Los Angeles Crime Dataset (Intimate Partner Assault, Stolen Vehicle, Plain Theft, Burglary, etc..) using **Random Forest Classifier**.

    #### Pages
    1. `Dataset` - Brief description and background of the Los Angeles Crime dataset used in this dashboard. 
    2. `EDA` - Exploratory Data Analysis of the Los Angeles Crime dataset. Highlighting the distribution of Crime Type and the relationship between the features. Includes graphs such as Pie Chart, .
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets.
    4. `Machine Learning` - Training a supervised classification model: Random Forrest Classifier. Includes model evaluation and highlights the feature importance.
    5. `Prediction` - Prediction page where users can input values to predict the Crime Type using the trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.


    """)
    

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("""

    The **Los Angeles Crime Dataset** was collected and designed by user 'SAVORSAUCE'. This dataset encompasses all recorded crime data in the city of Los Angeles from 2020 to the present day (2023). 
    
    It includes detailed information such as the time each crime was reported, the type of offense committed, and the geographical coordinates (longitude and latitude) of each incident location.
    
    **Content**
      
    This dataset consists of **752911** rows along with a number of **28 data columns** that contains the related data of the crime. The columns are as follows: **Crime Number**, **Date Reported*, **Date of Occurence**, **Time of Occurence**, **Area of Occurence**, **Crime Code**, **Victim Description**, and etc..

    `Link:` https://www.kaggle.com/datasets/nathaniellybrand/los-angeles-crime-dataset-2020-present           
                
    """)
    
    col_pic = st.columns((4), gap='medium')
    
    resize_dimensions = (500, 300)
    
    with col_pic[0]:
        crime_pic = Image.open('assets/crime_pictures/crime_1.jpg')
        crime_pic = crime_pic.resize(resize_dimensions)
        st.image(crime_pic)
        
    with col_pic[1]:
        crime_pic = Image.open('assets/crime_pictures/crime_2.jpg')
        crime_pic = crime_pic.resize(resize_dimensions)
        st.image(crime_pic)
        
    with col_pic[2]:
        crime_pic = Image.open('assets/crime_pictures/crime_3.jpg')
        crime_pic = crime_pic.resize(resize_dimensions)
        st.image(crime_pic)
    
    with col_pic[3]:
        crime_pic = Image.open('assets/crime_pictures/crime_4.jpg')
        crime_pic = crime_pic.resize(resize_dimensions)
        st.image(crime_pic)
        
    # Display the dataset
    
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(crime_df.head(), use_container_width=True, hide_index=True)
    
    # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(crime_df.describe(), use_container_width=True)
    
    st.markdown("""

    The results from `df.describe()` highlights the descriptive statistics about the dataset. The dataset includes 752,911 crime entries, detailing various aspects like the occurrence time, area codes, crime types, victim demographics, weapon use, and geographical coordinates. 
    
    The typical time of occurrence is approximately 1:34 PM, with area codes varying from 1 to 21. The average age of victims is 29.9 years, but negative values suggest possible data problems. Crime and premise codes display a broad array of categories, illustrating various types of incidents. Data on weapon usage is insufficient, encompassing merely around one-third of the total records. 
    
    Geospatial information indicates an emphasis on Southern California, probably Los Angeles, but the absence of values in latitude and longitude fields might reflect incomplete data. In general, the dataset seems diverse yet has inconsistencies that might need to be addressed before analysis.
    
    """)
    
    
    
# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((3, 3.5, 3), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Crime Area Distribution')
        bar_graph(crime_df, 'AREA NAME', 500, 300, 1)

    with col[1]:
        st.markdown('#### Crime Type Distribution')
        pie_chart_CrmCdDesc(crime_df, 500, 500, 2)
        
    with col[2]:
        st.markdown('#### Victum Age Distribution')
        plot_age_group_distribution(crime_df)
        
    st.markdown("---")
    
    st.header("💡 Insights")
    
    st.markdown("""

    As we can see from the pie chart above, the most committed crime in the dataset is VEHICLE - STOLEN with 10.7%. Next are BATTERY - SIMPLE ASSAULT with 7.9%, THEFT OF IDENTITY with 6.5%, BURGLARY FROM VEHICLE with 6.2%, and so on. As for the other crime types that are few in numbers, we will have to remove rows with these types in order to prevent biases in training the model due to the unbalanced dataset. This also means we will have to remove some rows with crime types that have high counts in the dataset. 
    
    The bar graph depicts the spread of criminal activities in various regions, emphasizing notable differences in crime occurrence. The Central region reports the highest crime rate, surpassing 50,000 cases, with the 77th Street and Pacific regions following closely behind. Regions such as Hollywood, Southwest, and Southeast likewise exhibit fairly elevated crime numbers, varying from 40,000 to 45,000. 
    
    In comparison, Foothill and Hollenbeck show the least crime figures, both below 30,000. The pattern indicates that crime rates tend to be elevated in central and densely populated areas, whereas peripheral regions encounter a lower number of incidents. 
    
    The other bar graph illustrate the distribution of crime victims in Los Angeles based on age demographics. The data reveals that the highest concentration of victims falls within the 31-45 age range, with approximately 200.6k individuals affected. This age group faces the highest number of crimes, indicating a peak vulnerability period. This is followed by the 18-30 age group, with a total of 153.8k victims, and then the 46-60 age group, which ranks below.
    
    The chart illustrates crime patterns among various age categories. Although it shows a distinct trend of rising victimization as age increases, then decreasing, more analysis is required to comprehend the underlying reasons and create effective crime prevention measures.

    """)


# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")
    
    st.dataframe(crime_df.head(), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    buffer = io.StringIO()
    crime_df.info(buf=buffer)
    summary_text = buffer.getvalue()
    
    col = st.columns((2), gap='large')
    
    with col[0]:
        st.markdown('##### Dataset Info Summary')
        st.text(summary_text)
    
    with col[1]:
        st.markdown('##### Null Values and Data Types')
        st.write("###### Number of Rows:", crime_df.shape[0])
        st.write("###### Number of Columns:", crime_df.shape[1])
        st.write("###### Data Types:", crime_df.dtypes, use_container_width=True)
        st.dataframe(crime_df.isnull().sum(), use_container_width=True)
        

    st.markdown("---")
    
    st.markdown("""
                
    We can see from the table that there are significantly many null values across many columns. These columns are Mocodes, Weapon Used Cd, Weapon Desc, Crm Cd 2, Crm Cd 3, Crm Cd 4, and Cross Street. We will have to drop them in a new data frame as well as the deemed unnecessary columns in predicting the types of crime committed such as DR_NO, Date Rptd, Rpt Dist No, Status, Status Desc, Crm Cd 1, LOCATION, LAT, and LON.

    Vict Sex, Vict Descent, Premis Cd, and Premis Desc are necessary. And so, we will try to reduce the number of their null values.

    """)
    
    col1 = st.columns((3), gap='small')
    
    with col1[0]:
        st.markdown('##### Victim Age Column')
        st.dataframe(crime_df['Vict Age'].value_counts(), use_container_width=True)
        
    with col1[1]:
        st.markdown('##### Victim Sex Column')
        st.dataframe(crime_df['Vict Sex'].value_counts(), use_container_width=True)
        
    with col1[2]:
        st.markdown('##### Victim Descent Column')
        st.dataframe(crime_df['Vict Descent'].value_counts(), use_container_width=True)
        
    st.markdown("---")
    
    st.markdown("""
                
   In the Victim Sex and Victim Descent Columns, we have to replace the null values with '-' because the value '-' exist in these columns and are deemed to represent crimes where there are no victims involved. In addition to that, the ages -1 and -2 in the Victim Age Column do not make sense and therefore should be interpreted the same as 0.

    """)
    
    st.dataframe(crime_df[['Premis Cd', 'Premis Desc']].value_counts(), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("""
                
   On the other hand, Premis Cd and its corresponding column Premis Desc have null values that do not represent anything and are in few numbers. And so, we will have to drop the rows containing null values in said columns. We will also have to drop the row containing 803.0 and 805.0 in Premis Cd since its literal description is RETIRED (DUPLICATE) DO NOT USE THIS CODE.
   
   Using the function clean_data(), we'll be able to remove and replace unnessecary values to make our dataset more accurate once it goes through the machine learning portion of the process.

    """)
    
    crime_df_cleaned = clean_data(crime_df)
    
    st.code('crime_df_cleaned = clean_data(crime_df)')
    
    st.dataframe(crime_df_cleaned.head(), use_container_width=True)
    
    st.header('Data Balancing')
    
    # Make 'Crm Cd Desc' the target variable
    CrmDesc_counts = crime_df_cleaned['Crm Cd Desc'].value_counts()

    # Filter out classes with a small number of occurrences
    threshold_percentage = 4     # Adjust as desired
    total_count = len(crime_df_cleaned)
    threshold_count = total_count * (threshold_percentage / 100)

    significant_classes = CrmDesc_counts[CrmDesc_counts >= threshold_count].index.tolist()

    st.write(f"Original DataFrame size: {len(crime_df_cleaned)}")

    # Filter the DataFrame to include only rows with significant classes
    crime_df_cleaned = crime_df_cleaned[crime_df_cleaned['Crm Cd Desc'].isin(significant_classes)]

    st.write(f"Filtered DataFrame size: {len(crime_df_cleaned)}")

    # Now 'crime_df_cleaned' contains only rows with the most significant crime types.
    # We can proceed with further cleaning, preprocessing, and model training using 'crime_df_cleaned'.

    crime_df_cleaned.head()

    if 'Crm Cd Desc' in crime_df_cleaned.columns:
        CrmDesc_counts = crime_df_cleaned['Crm Cd Desc'].value_counts()
        st.write("Class Distribution:\n", CrmDesc_counts)
        
    st.markdown("""
                
   To ensure the dataset is well-suited for model training and testing, we must first balance it by excluding values in the "Crime Code Description" (Crm Cd Desc) and filtering out less significant crime types. This preprocessing step will help the model train more efficiently, minimizing the risk of overfitting or struggling with irrelevant data.

    """)
    
    pie_chart_CrmCdDesc(crime_df_cleaned, 500, 500, 1, top_n=10)
    
    col2 = st.columns((2), gap='medium')

    # Display original class distribution
    original_counts = crime_df_cleaned['Crm Cd Desc'].value_counts()
    
    with col2[0]:
        st.markdown('#### Original Class Distribution:')
        st.write(original_counts)
        
    # Balance the dataset
    balanced_df = balance_dataset(crime_df_cleaned, 'Crm Cd Desc')

    # Display balanced class distribution
    balanced_counts = balanced_df['Crm Cd Desc'].value_counts()
    
    with col2[1]:
        st.markdown('#### Balanced Class Distribution:')
        st.write(balanced_counts)
        
    st.markdown("""
                
   To further optimize the dataset for model training and testing, it is essential to resample and balance the values in the "Crime Code Description" (Crm Cd Desc), which serves as the target variable for our classification model. This approach will enhance the model's ability to make accurate predictions, while mitigating the risk of overfitting caused by imbalanced data, thereby improving its overall performance and reliability.
   
   With the distribution of crime types in the dataset now **balanced** and null values successfully removed, we can proceed to generate embeddings for the *Crm Cd Desc* column and perform the Train-Test split to prepare for the training of our machine learning model.

    """)
    
    col3 = st.columns((3), gap='medium')
    
    with col3[0]:
        st.markdown('#### Premise Code Column:')
        st.dataframe(crime_df_cleaned['Premis Cd'], use_container_width=True, hide_index=True)
        
    with col3[1]:
        st.markdown('#### Date of Occurence Column:')
        st.dataframe(crime_df_cleaned['DATE OCC'], use_container_width=True, hide_index=True)
        
    with col3[2]:
        st.markdown('#### Time of Occurence Column:')
        st.dataframe(crime_df_cleaned['TIME OCC'], use_container_width=True, hide_index=True)
        
    st.markdown("---")
    
    st.markdown("""
                
    As indicated in the columns above, both "Premise Code" (Premis Cd) and "Date of Occurrence" (DATE OCC) are not in their correct formats. To address this issue, we need to convert the "Premise Code" from a float to an integer data type, "Time of Occurence" values to strings to extract the hour values, and transform the "Date of Occurrence" from an object type to a datetime format.   
     
    """)
    
    crime_df_cleaned['Premis Cd'] = crime_df_cleaned['Premis Cd'].astype(int)
    
    crime_df_cleaned['DATE OCC'] = pd.to_datetime(crime_df_cleaned['DATE OCC'], errors='coerce')
    
    crime_df_cleaned['TIME OCC'] = crime_df_cleaned['TIME OCC'].astype(str).str.zfill(4)
    
    # Check for NaT values in 'DATE OCC' and 'TIME OCC'
    nat_rows_date = crime_df_cleaned[crime_df_cleaned['DATE OCC'].isna()]
    nat_rows_time = crime_df_cleaned[crime_df_cleaned['TIME OCC'].isna()]
    
    # Create new features from the date
    crime_df_cleaned['Month'] = crime_df_cleaned['DATE OCC'].dt.month
    crime_df_cleaned['Hour'] = crime_df_cleaned['TIME OCC'].str[:2].astype(int)
    
    # Mapping of the Crm Cd Desc and their encoded equivalent
    unique_CrmCdDesc = crime_df_cleaned['Crm Cd Desc'].unique()
    unique_CrmCd = crime_df_cleaned['Crm Cd'].unique()

    # Create a new DataFrame
    CrmCdDesc_mapping_df = pd.DataFrame({'Crm Cd Desc': unique_CrmCdDesc, 'Crm Cd': unique_CrmCd})
    
    # Mapping of the AREA NAME and their encoded equivalent

    unique_AreaName = crime_df_cleaned['AREA NAME'].unique()
    unique_Area = crime_df_cleaned['AREA'].unique()

    # Create a new DataFrame
    AreaName_mapping_df = pd.DataFrame({'AREA NAME': unique_AreaName, 'AREA': unique_Area})  
    
    
    # Mapping of the Premis Desc and their encoded equivalent
    unique_PremisDesc = crime_df_cleaned['Premis Desc'].unique()
    unique_PremisCd = crime_df_cleaned['Premis Cd'].unique()

    # Create a new DataFrame
    PremisDesc_mapping_df = pd.DataFrame({'Premis Desc': unique_PremisDesc, 'Premis Cd': unique_PremisCd})
    
    col4 = st.columns((3), gap='medium')
    
    with col4[0]:
        st.markdown('#### Mapped Crime Code:')
        st.dataframe(CrmCdDesc_mapping_df, use_container_width=True, hide_index=True)
    
    with col4[1]:
        st.markdown('#### Mapped Area Code:')
        st.dataframe(AreaName_mapping_df, use_container_width=True, hide_index=True)
        
    with col4[2]:
        st.markdown('#### Mapped Premise Code:')
        st.dataframe(PremisDesc_mapping_df, use_container_width=True, hide_index=True)
        
    st.markdown("---")
        
    st.markdown("""
             
    In the column above, we have mapped the unique values of Crime Code (Crm Cd), Premise Code (Premis Cd), and Area Code (AREA) to their respective Crime Code (Crm Cd Desc), Premise Code (Premis Cd Desc) and Area Code (AREA NAME) descriptions.   
     
    """)
    
    st.code('encoder = LabelEncoder()')
    encoder = LabelEncoder()
    
    crime_df_cleaned['Vict Sex Encoded'] = encoder.fit_transform(crime_df_cleaned['Vict Sex'])
    
    crime_df_cleaned['Vict Descent Encoded'] = encoder.fit_transform(crime_df_cleaned['Vict Descent'])
    
    # Mapping of the Vict Sex and their encoded equivalent
    unique_VictSex = crime_df_cleaned['Vict Sex'].unique()
    unique_VictSexEncoded = crime_df_cleaned['Vict Sex Encoded'].unique()

    # Create a new DataFrame
    VictSex_mapping_df = pd.DataFrame({'Vict Sex': unique_VictSex, 'Vict Sex Encoded': unique_VictSexEncoded})
    
    # Mapping of the Vict Descent and their encoded equivalent

    unique_VictDescent = crime_df_cleaned['Vict Descent'].unique()
    unique_VictDescentEncoded = crime_df_cleaned['Vict Descent Encoded'].unique()

    # Create a new DataFrame
    VictDescent_mapping_df = pd.DataFrame({'Vict Descent': unique_VictDescent, 'Vict Descent Encoded': unique_VictDescentEncoded})
    
    col5 = st.columns((2), gap='medium')
    
    with col5[0]:
        st.markdown('#### Encoded and Mapped Victim Sex:')
        st.dataframe(VictSex_mapping_df.head(), use_container_width=True, hide_index=True)
    
    with col5[1]:
        st.markdown('#### Encoded and Mapped Victim Descent:')
        st.dataframe(VictDescent_mapping_df.head(), use_container_width=True, hide_index=True)
        
    st.markdown("---")
    
    st.markdown("""
             
    After initializing the LabelEncoder and mapping the existing columns to their respective codes, we now need to generate encodings for columns that lack predefined labels. For instance, attributes like "Victim Sex" and "Victim Descent" currently have no associated encodings. Hence, we use the LabelEncoder to create the necessary labels for these columns.     
    
    """)
    
    # Select features and target variable
    features = ['AREA', 'Part 1-2', 'Vict Age', 'Premis Cd', 'Month',
                'Hour', 'Vict Sex Encoded', 'Vict Descent Encoded']
    X = crime_df_cleaned[features]
    y = crime_df_cleaned['Crm Cd']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    code_block = """
    features = ['AREA', 'Part 1-2', 'Vict Age', 'Premis Cd', 'Month', 'Hour', 'Vict Sex Encoded', 'Vict Descent Encoded']
    X = crime_df_cleaned[features]
    y = crime_df_cleaned['Crm Cd']
    """

    st.code(code_block, language='python')
    
    st.code('X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)')
    
    col6 = st.columns((2), gap='medium')
    
    with col6[0]:
        st.markdown('#### X and Y Head Columns:')
        col7 = st.columns((2), gap='medium')
        
        with col7[0]:
            st.markdown('#### X Head')
            st.dataframe(X, use_container_width=True, hide_index=True)
            
        with col7[1]:
            st.markdown('#### Y Head')
            st.dataframe(y, use_container_width=True, hide_index=True)
            
    with col6[1]:
        st.markdown('#### X Train and Y Train Head Columns:')
        col8 = st.columns((2), gap='medium')
        
        with col8[0]:
            st.markdown('#### X Train Head')
            st.dataframe(X_train, use_container_width=True, hide_index=True)
            
        with col8[1]:
            st.markdown('#### Y Train Head')
            st.dataframe(y_train, use_container_width=True, hide_index=True)
            
    st.markdown("---")
            
    st.markdown("""
            
    In this portion of the Data Pre-Processing, we define the features and the target variable that are going to be used in the Machine Learning portion and split the dataset into their training and testing portion. Dividing the dataset into training and testing groups is essential for creating successful machine learning models. It enables an unbiased assessment on unknown data, aiding in evaluating the model's generalization and minimizing the chances of overfitting. This division guarantees impartial validation and aids in efficient hyperparameter tuning, resulting in a more trustworthy and resilient model that excels in real-world situations.      
    
    """)
        

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    st.subheader("Random Forest Classifier")
    
    st.markdown("""
            
    The Random Forest Classifier is a popular machine learning model used for sorting data into categories. It works by creating many decision trees, each trained on a different random sample of the data. To make things even more random, each tree only considers a random subset of features at each decision point. The final prediction is determined by a majority vote, where the most frequent outcome from all the trees is selected. This method is effective because it reduces overfitting and improves accuracy.
    
    `Reference:` https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    
    """)
    
    col_pic = st.columns((2, 4, 2), gap='medium')
    
    with col_pic[0]:
        st.write(' ')

    with col_pic[1]:
        rfrimage = Image.open('assets/figure/randomforestclassifier.jpg')
        st.image(rfrimage)

    with col_pic[2]:
        st.write(' ')
    
    st.subheader("Training the Random Forest Classifier")
    
    st.code("""

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)     
            
    """)
    
    st.code("""

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')    
            
    """)
    
    st.write("Accuracy: 80.75%")
    
    st.markdown("""

    After training our Random Forest classifier, the model achieved an accuracy of 80.75%, showing its ability to learn and make correct predictions. However, despite efforts to balance and clean the dataset, certain issues within the data still led to inaccuracies that reduced the overall model performance.     
    
    """)
    
    st.subheader("Feature Importance")
    
    feature_importance_data = {
    'Feature': ['AREA', 'Part 1-2', 'Vict Age', 'Premis Cd', 'Month', 
                'Hour', 'Vict Sex Encoded', 'Vict Descent Encoded'],
    'Importance': [0.109690, 0.146127, 0.187525, 0.193335, 0.095207, 
                   0.108329, 0.063119, 0.096668]
    }
    
    feature_importance_df = pd.DataFrame(feature_importance_data)
    st.dataframe(feature_importance_df)
    
    feature_importance_plot(feature_importance_df, 500, 500, 1)
    
    st.subheader("Number of Trees")
    st.code("""

    print(f"Number of trees made: {len(model.estimators_)}")
     
    """)
    
elif  st.session_state.page_selection == "prediction":
    st.header(" Prediction")

    # Initialize session state for clearing results
    if 'clear' not in st.session_state:
        st.session_state.clear = False
        
    st.markdown("#### 🌳 Random Forest Classifier")
    
    col_pred = st.columns((2), gap='medium')
    
    area_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    part_options = ['1', '2']
    vict_sex_list = ['0', '1', '2', '3', '4']
    premis_cd_list = [
        '502', '707', '101', '501', '108', '723', '717', '122', '714', '102', '203', '503', '145', '733', '403', '104',
        '121', '406', '124', '242', '252', '602', '509', '116', '410', '109', '405', '750', '119', '415', '222', '247',
        '510', '117', '514', '210', '404', '905', '221', '202', '248', '103', '701', '118', '504', '702', '708', '402',
        '213', '217', '722', '301', '605', '753', '752', '720', '212', '505', '909', '401', '211', '710', '801', '123',
        '721', '725', '507', '244', '738', '208', '719', '704', '255', '518', '744', '413', '727', '516', '745', '120',
        '408', '932', '506', '604', '128', '112', '417', '158', '243', '207', '603', '810', '726', '729', '951', '742',
        '732', '834', '146', '407', '736', '711', '110', '936', '876', '144', '229', '220', '751', '250', '735', '517',
        '251', '730', '903', '912', '303', '728', '228', '515', '218', '913', '917', '705', '150', '139', '601', '107',
        '148', '740', '232', '201', '508', '922', '114', '138', '709', '512', '713', '911', '239', '156', '902', '142',
        '835', '716', '234', '511', '245', '106', '219', '152', '409', '931', '957', '154', '236', '921', '724', '231',
        '754', '904', '703', '970', '254', '943', '411', '151', '910', '141', '900', '706', '223', '149', '907', '205',
        '607', '802', '143', '958', '948', '737', '302', '519', '235', '940', '937', '209', '147', '157', '140', '253',
        '249', '908', '804', '920', '214', '412', '945', '135', '712', '944', '883', '882', '962', '216', '111', '741',
        '871', '233', '155', '967', '755', '241', '238', '416', '919', '204', '971', '916', '933', '906', '809', '304',
        '897', '901', '874', '946', '966', '237', '129', '934', '947', '513', '718', '811', '757', '230', '918', '963',
        '246', '739', '224', '893', '950', '414', '875', '885', '949', '115', '126', '956', '731', '872', '748', '206',
        '868', '954', '105', '127', '961', '952', '935', '136', '873', '964', '225', '877', '941', '879', '895', '606',
        '227', '880', '836', '953', '942', '758', '969', '896', '215', '608', '869', '113', '756', '894', '734', '968',
        '125', '892', '743'
    ]    
    
    vict_desc_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']   
    
    # Input boxes for the features:
    
    with col_pred[0]:
        area_selection = st.selectbox(
        label='Area',
        options=area_list,
        index=0,  
        key='area_selection', 
        help="Choose an Area"  
        )
    
        crime_part_selection = st.selectbox(
        label='Part 1-2',
        options= part_options,
        index= 0,
        key= 'crime_part_selection',
        help= "Choose between Part 1 or 2"
        )
    
        vict_age = st.number_input(
        label= 'Victim Age', 
        min_value=0, 
        max_value=120, 
        step=1, 
        key='rfc_vict_age', 
        value=0 
        if st.session_state.clear 
        else st.session_state.get('rfc_vict_age', 0)
        )
    
        premis_cd_selection = st.selectbox(
        label='Premise Code',
        options=premis_cd_list,
        key='premis_cd_selection',
        index=premis_cd_list.index(st.session_state.get('rfc_premis_cd', 0)) if st.session_state.get('rfc_premis_cd', 0) in premis_cd_list else 0,
        help='Select a premise code from the list. (eg. 127)'
        )
        
    with col_pred[1]:
        rfc_month = st.number_input(
        label='Month', 
        min_value=1, 
        max_value=12, 
        step=1, key='rfc_month', 
        value=1 
        if st.session_state.clear 
        else st.session_state.get('rfc_month', 1)
        )
    
        rfc_hour = st.number_input(
        label='Hour', 
        min_value=0, 
        max_value=23, 
        step=1, 
        key='rfc_hour', 
        value=0 
        if st.session_state.clear 
        else st.session_state.get('rfc_hour', 0)
        )
    
        rfc_vict_sex_enc = st.selectbox(
        label='Victim Sex Encoded',
        options= vict_sex_list,
        key='rfc_vict_sex_enc',
        index=0 if st.session_state.clear else vict_sex_list.index(st.session_state.get('rfc_vict_sex_enc', '0')),
        help='Select a Sex (eg. 1 = F, 2 = M, etc.)'
        )
    
        rfc_vict_desc_enc = st.selectbox(
        label='Victim Descent Encoded',
        options= vict_desc_list,
        key='rfc_vict_desc_enc',
        index=0 if st.session_state.clear else vict_sex_list.index(st.session_state.get('rfc_vict_desc_enc', '0')),
        help='Select a Descent (eg. 1 = F, 2 = M, etc.)'
        )

    crime_cd_list = ['626', '510', '440', '310', '354', '230', '330', '740', '624']
    
    crime_cd_desc_dict = { 
    '624': 'BATTERY-SIMPLE ASSAULT',
    '510': 'VEHICLE-STOLEN',
    '440': 'THEFT PLAIN PETTY ($950 & UNDER)',
    '310': 'BURGLARY',
    '354': 'THEFT OF IDENTITY',
    '230': 'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',
    '330': 'BURGLARY FROM VEHICLE',
    '740': 'VANDALISM-FELONY ($400 & OVER, ALL CHURCH VANDALISMS)',
    '626': 'INTIMATE PARTNER-SIMPLE ASSAULT'
    }
    
    # Button to detect the Crime Type
    if st.button('Detect', key='rfc_detect'):
        # Prepare the input data for prediction
        rfc_input_data = [[area_selection , crime_part_selection, vict_age , premis_cd_selection, rfc_month, rfc_hour, rfc_vict_sex_enc, rfc_vict_desc_enc]]
        
        # Predict the Crime Type
        rfc_prediction = model.predict(rfc_input_data)

        predicted_crime_code = str(rfc_prediction[0])
        predicted_crime_desc = crime_cd_desc_dict.get(predicted_crime_code, "Unknown Crime")
        
        st.markdown(f"The predicted Crime Type is: **{predicted_crime_desc}** (`{predicted_crime_code}`)")

elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
    
    st.markdown("""
                
        In this analysis, we focused on predicting crime types using a Random Forest Classifier model. We began with an unbalanced dataset containing **752,911** records and removed rows representing crime types that made up less than **4%** each of the total data. After this data cleaning and balancing process, the dataset was reduced to **341,505** rows, concentrating on the top nine crime types. This selection made us worked with the most frequently recorded crime types, including "`INTIMATE PARTNER - SIMPLE ASSAULT`", "`VEHICLE - STOLEN`", "`THEFT PLAIN - PETTY ($950 & UNDER)`", "`BURGLARY`", "`THEFT OF IDENTITY`", "`ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT`", "`BURGLARY FROM VEHICLE`", "`VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)`", and "`BATTERY - SIMPLE ASSAULT`"


        #### 1.   Model Training and Performance:

        - We trained our `Random Forest Classifier` on selected features such as `AREA NAME`, `Part 1-2`, `Vict Age`, `Premis Desc`, `Month`, `Hour`, `Vict Sex`, and `Vict Descent`  to predict the crime category. After splitting the data, the model achieved an accuracy of **80.75%** on the test set. This performance indicates that our selected features provided substantial predictive power, though additional cleaning and balancing of data could potentially improve accuracy. But also means sacrificing another type of crime in the model training just so the accuracy would improve, and so we chose not to.

        #### 2.   Feature Importance Analysis:

        -  Analysis of feature importance within the `Random Forest Classifier` model highlighted key predictors: `Premis Cd` (location type) and `Vict Age` were the most influential features, contributing around 19% and 18.7% respectively to the model's predictions. Other important factors included `Part 1-2` (crime part category) and `AREA NAME`. The prominence of these features show the value of contextual and demographic data in crime prediction. Unexpectedly, `Vict Sex` had the lowest importance among all the features. Other features were neither highly important nor not invaluable enough to be considered removed from the model training.


        ### Summary

        Overall, this project demonstrated the viability of using a machine learning approach to predict crime types with a reasonable degree of **80.75%** accuracy. By focusing on a balanced subset of common crime types and the use of a Random Forest Classifier model, we were able to successfully find patterns in the data that correlate with different crime categories. Though many crime types have been excluded, this project could definitely be improved with the help of other model training tools.
                
    """)

            
            
            
    

        
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
        
    
        
    
    
    













