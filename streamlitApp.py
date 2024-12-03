import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load data (replace with your data loading logic)
# @st.cache
def load_data():
    return pd.read_csv("HRDataset_v14.csv")

df = load_data()

# Convert relevant columns to datetime

df['DateofHire'] = pd.to_datetime(df['DateofHire'])
df['DateofTermination'] = pd.to_datetime(df['DateofTermination'])

df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%y')
df['DOB'] = df['DOB'].apply(lambda x: x - pd.DateOffset(years=100) if x.year > 2022 else x)
df['Age'] = (pd.to_datetime('today') - df['DOB']).dt.days // 365 

df = df[df['Salary'] < df['Salary'].quantile(0.99)]

bins = [20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80]
labels = ['20-22', '23-25', '26-28', '29-31', '32-34', '35-37', '38-40', '41-43', '44-46', '47-49', 
          '50-52', '53-55', '56-58', '59-61', '62-64', '65-67', '68-70', '71-73', '74-76', '77-79']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)


def dashboard():
    # Styling improvements
    st.markdown(
        """
        <style>
        .st-emotion-cache-6qob1r {background-color: #ebe6f2;}
        .st-emotion-cache-1jicfl2 {
                        padding-top: 3rem;
                        padding-bottom: 1rem;
                        padding-left: 20px;
                        padding-right: 20px;
                    }
        .css-1d391kg {
            padding-top: 3rem;
            padding-right: 1rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

     # Metrics for salary equality
    st.sidebar.header("Salary Equality Metrics")

    # Add filters for interactivity
    st.sidebar.header("Filters")
    selected_department = st.sidebar.selectbox("Department", options=['All'] + list(df['Department'].unique()), index=0)
    if selected_department is 'All':
        filtered_data = df
    else:
        filtered_data = df[df['Department'] == selected_department]

    st.markdown('<h1 style = "color:#735DA5; padding-bottom: 0px">Employee Salary Equality Dashboard</h1>', unsafe_allow_html=True)
    subheading = f"Salary Data for Department: {selected_department}"
    st.markdown(f'<h3 style = "padding-bottom: 0px">{subheading}</h3>', unsafe_allow_html=True)
    

    age_salary = filtered_data.groupby('AgeGroup')['Salary'].describe().round(2)

    race_salary = filtered_data.groupby('RaceDesc')['Salary'].describe().round(2)

    gender_salary = filtered_data.groupby('Sex')['Salary'].describe().round(2)
    gender_avg_salary = filtered_data.groupby('Sex', as_index=False)['Salary'].mean()


    gender_gap = gender_salary['mean'][1] - gender_salary['mean'][0]
    st.sidebar.metric("Gender Salary Gap (M - F)", f"${gender_gap:,.2f}")

    race_salary_diff = race_salary['mean'].max() - race_salary['mean'].min()
    st.sidebar.metric("Race Salary Disparity", f"${race_salary_diff:,.2f}")

    max_salary_row = race_salary.loc[race_salary['mean'].idxmax()]
    max_salary_race = max_salary_row.name  

    min_salary_row = race_salary.loc[race_salary['mean'].idxmin()]
    min_salary_race = min_salary_row.name  

    st.sidebar.caption(f"The race with the highest average salary is: {max_salary_race}")
    st.sidebar.caption(f"The race with the lowest average salary is: {min_salary_race}")
    
    st.sidebar.markdown(
        """
        ---
        **Developed by Luke Carroll using Streamlit and Plotly**
        [Click Here for my Website](https://lukecarrolldata.org)
        """
    )

    bg_colour = '#F2F1F1'

    st.markdown(
        f"""
        <style>
        .stPlotlyChart {{
            background-color: {bg_colour};
            outline: 5px solid {bg_colour};
            border-radius: 5px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.05), 0 6px 20px 0 rgba(0, 0, 0, 0.15);
        }}

        .stContainer {{
            background-color: {bg_colour};
            outline: 5px solid {bg_colour};
            border-radius: 5px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.05), 0 6px 20px 0 rgba(0, 0, 0, 0.15);
        }}
        </style>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap = 'medium')

    with col1:
        fig_gender = px.bar(
            gender_avg_salary,
            x='Salary',
            y='Sex',
            template="plotly_white",
            labels={'Salary': 'Average Salary', 'Sex': 'Gender'},
            color = 'Sex',
            color_discrete_map={ 'F': '#735DA5', 'M ': '#D3C5E5',},
            
        )

        fig_gender.update_layout(
            paper_bgcolor = bg_colour,
            plot_bgcolor = bg_colour,
            title_text = "Salary by Gender",
            height = 300,
            legend=dict(
                orientation="h", 
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_gender, config={"displayModeBar": False}) # use_container_width=True, 
        


    with col2:
        gender_race_salary = (
            filtered_data.groupby(['Sex', 'RaceDesc'])['Salary']
            .mean()
            .reset_index()
        )

        def add_line_breaks(text):
            words = text.split(' ')
            return '<br>'.join([' '.join(words[i:i+2]) for i in range(0, len(words), 2)])

        gender_race_salary['RaceDesc'] = gender_race_salary['RaceDesc'].apply(add_line_breaks)

        fig_gender_race = px.bar(
            gender_race_salary,
            orientation ='h',
            x = 'Salary',
            y = 'RaceDesc',
            color = 'Sex',
            barmode = 'group',
            template = "plotly_white",
            labels={'RaceDesc': 'Race', 'Salary': 'Average Salary', 'Sex': 'Gender'},
            color_discrete_map={'M ': '#D3C5E5', 'F': '#735DA5'}
        )

        fig_gender_race.update_layout(
            paper_bgcolor = bg_colour,
            plot_bgcolor = bg_colour,
            title_text = "Salary by Gender and Race",
            height = 300,
            legend = dict(
                orientation = "h",  # Horizontal layout
                yanchor = "bottom",  # Align the bottom of the legend
                y = 1.02,  # Slightly above the plot
                xanchor = "right",  # Align the legend to the right
                x = 1  # Position at the far right horizontally
            )
        )

        st.plotly_chart(fig_gender_race, use_container_width=True, config={"displayModeBar": False})

        
    st.text('')

    df_avg_salary = filtered_data.groupby(['AgeGroup', 'Sex'], as_index=False)['Salary'].mean()

    fig_age = px.line(
        df_avg_salary, 
        x = 'AgeGroup', 
        y = 'Salary', 
        color = 'Sex',  
        template = "plotly_white",
        labels = {'AgeGroup': 'Age Group', 'Salary': 'Average Salary', 'Sex': 'Gender'},
        line_shape = 'spline',
        color_discrete_map = {'M ': '#D3C5E5', 'F': '#735DA5'},
        markers = True
    )

    fig_age.update_traces(
        fill='tozeroy',  
        fillcolor='rgba(115, 93, 165, 0.1)'
    )

    fig_age.update_layout(
        paper_bgcolor=bg_colour,
        plot_bgcolor=bg_colour,
        title_text="Average Salary by Gender and Age Group",
        height = 350
    )

    st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})

    st.text('')

def ML_Deployment():
    st.markdown(
        """
        <style>
        .st-emotion-cache-6qob1r {background-color: #ebe6f2;}
        .st-emotion-cache-1jicfl2 {
                        padding-top: 3rem;
                        padding-bottom: 1rem;
                        padding-left: 20px;
                        padding-right: 20px;
                    }
        .css-1d391kg {
            padding-top: 3rem;
            padding-right: 1rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    st.markdown('<h1 style="color:#735DA5;">Machine Learning Model Predicting Employee Salary</h1>', unsafe_allow_html=True)


    # with open('model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    
    def get_user_input():
        department = st.selectbox("Department", options = df['Department'].unique(), index = 0)
        gender = st.selectbox("Gender", options = ["Other"] + list(df['Sex'].unique()), index = 0)
        age = st.number_input('Age', min_value=18)
        marital_status = st.selectbox('Marital Status', ['Divorced', 'Married', 'Separated', 'Single', 'Widowed'])

        department_dict = {f'Department_{dep}': int(dep == department) for dep in df['Department'].unique()}
        sex_dict = {f'Department_{gen}': int(gen == gender) for gen in ["Other"] + list(df['Sex'].unique())}

        user_input = pd.DataFrame({
            'Age': [age],
            **sex_dict,
            **department_dict,
            'MaritalDesc_Divorced': [1 if marital_status == 'Divorced' else 0],
            'MaritalDesc_Married': [1 if marital_status == 'Married' else 0],
            'MaritalDesc_Separated': [1 if marital_status == 'Separated' else 0],
            'MaritalDesc_Single': [1 if marital_status == 'Single' else 0],
            'MaritalDesc_Widowed': [1 if marital_status == 'Widowed' else 0],
        })

        return user_input
    
    def transform_input_data(user_input):
        user_input_transformed = pd.get_dummies(user_input)
        return user_input_transformed
    
    # def make_prediction(user_input):
    #     # Transform categorical data
    #     transformed_input = transform_input_data(user_input)
    #     # Predict using the trained model
    #     prediction = model.predict(transformed_input)
    #     return prediction
    
    # def display_results(prediction):
    #     st.write(f'The model prediction is: {prediction[0]}')

    st.write('COMING SOON- Please fill in the details below to get the prediction.')

    user_input = get_user_input()

    # prediction = make_prediction(user_input)

    # display_results(prediction)



st.set_page_config(page_title="Employee Salary", page_icon="💼", layout="wide")


page_names_to_funcs = {
    "Dashboard": dashboard,
    "Machine Learning Model": ML_Deployment
}

demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()