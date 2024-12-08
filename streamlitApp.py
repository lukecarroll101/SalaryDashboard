import streamlit as st
import pandas as pd
# import seaborn as sns
import plotly.express as px
import joblib

# Load data (replace with your data loading logic)
# @st.cache
def load_data():
    return pd.read_csv("salary_with_department.csv")

df = load_data()


age_col = 'Age'
race_col = 'Race'
salary_col = 'Salary'
gender_col = 'Gender'

primary_colour = '#'
secondary_colour = '#'
sidebar_colour = '#ebe6f2' # purple: '#ebe6f2' green: '#eaf2e6'

male_colour = '#D3C5E5' # purple: '#D3C5E5' green: '#bee6ac'
female_colour = '#735DA5' # purple: '#735DA5' green: '#6ac93a'

bins = [ 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65]
labels = ['17-19', '20-22', '23-25', '26-28', '29-31', '32-34', '35-37', '38-40', '41-43', '44-46', '47-49', 
          '50-52', '53-55', '56-58', '59-61', '62-64']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

def dashboard():
    # Styling improvements
    st.markdown(
        f"""
        <style>
        .st-emotion-cache-6qob1r {{
            background-color: {sidebar_colour};
        }}
        .st-emotion-cache-1jicfl2 {{
                        padding-top: 3rem;
                        padding-bottom: 1rem;
                        padding-left: 20px;
                        padding-right: 20px;
                    }}
        .css-1d391kg {{
            padding-top: 3rem;
            padding-right: 1rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add filters for interactivity
    st.sidebar.header("Filters")
    options = ['All'] + [dept for dept in df['Department'].unique() if dept not in ['Management', 'Administration']]
    selected_department = st.sidebar.selectbox("Department", options = options, index=0)
    if selected_department == 'All':
        filtered_data = df
    else:
        filtered_data = df[df['Department'] == selected_department]

    st.markdown(f'<h1 style = "color:{female_colour}; padding-bottom: 0px">Employee Salary Equality Dashboard</h1>', unsafe_allow_html=True)
    subheading = f"Salary Data for Department: {selected_department}"
    st.markdown(f'<h3 style = "padding-bottom: 0px">{subheading}</h3>', unsafe_allow_html=True)
    

    age_salary = filtered_data.groupby('AgeGroup')['Salary'].describe().round(2)

    race_salary = filtered_data.groupby('Race')['Salary'].describe().round(2)

    gender_salary = filtered_data.groupby('Gender')['Salary'].describe().round(2)
    gender_avg_salary = filtered_data.groupby('Gender', as_index=False)['Salary'].mean()

    gender_race_salary = (
    filtered_data.groupby(['Gender', 'Race'])['Salary']
    .mean()
    .reset_index()
        )

    st.sidebar.header("Salary Equality Metrics")

    gender_gap = gender_salary['mean'][1] - gender_salary['mean'][0]
    st.sidebar.metric("Gender Salary Gap (M - F)", f"${gender_gap:,.2f}")

    side_col1, side_col2 = st.sidebar.columns(2) 

    side_col1.metric( "Females",f"{filtered_data['Gender'].value_counts().get('Female', ):,}")
    side_col2.metric( "Males",f"{filtered_data['Gender'].value_counts().get('Male', ):,}")
    
    max_salary_row = race_salary.loc[race_salary['mean'].idxmax()]
    max_salary_race = max_salary_row.name  

    min_salary_row = race_salary.loc[race_salary['mean'].idxmin()]
    min_salary_race = min_salary_row.name  

    

    # st.sidebar.text(f"The race with the highest average salary is: {max_salary_race}")
    # st.sidebar.text(f"The race with the lowest average salary is: {min_salary_race}")
    
    st.sidebar.markdown(
        """
        ---
        **Developed by Luke Carroll using Streamlit and Plotly**

        [Click Here for Luke's Website](https://lukecarrolldata.org)
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
        </style>
        """, unsafe_allow_html=True)
    

    col1, col2 = st.columns(2, gap = 'medium')

    with col1:
        fig_gender = px.bar(
            gender_avg_salary[gender_avg_salary['Gender'] != 'Other'],
            x='Salary',
            y='Gender',
            template="plotly_white",
            labels={'Salary': 'Average Salary'},
            color = 'Gender',
            color_discrete_map={ 'Female': female_colour, 'Male': male_colour,},
            
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
        
        st.plotly_chart(fig_gender, config={"displayModeBar": False})
        


    with col2:
        fig_gender_race = px.bar(
            gender_race_salary[gender_race_salary['Gender'] != "Other"],
            x = 'Salary',
            y = 'Race',
            color = 'Gender',
            barmode = 'group',
            template = "plotly_white",
            labels={'Salary': 'Average Salary'},
            color_discrete_map={'Female': female_colour, 'Male': male_colour}
        )

        fig_gender_race.update_layout(
            paper_bgcolor = bg_colour,
            plot_bgcolor = bg_colour,
            title_text = "Salary by Gender and Race",
            height = 300,
            yaxis=dict(autorange="reversed"),
            legend = dict(
                orientation = "h",  
                yanchor = "bottom", 
                y = 1.02,  
                xanchor = "right",  
                x = 1 
            )
        )

        st.plotly_chart(fig_gender_race, use_container_width=True, config={"displayModeBar": False})

        
    st.text('')

    df_avg_salary = filtered_data.groupby(['AgeGroup', 'Gender'], as_index=False)['Salary'].mean()

    fig_age = px.line(
        df_avg_salary[df_avg_salary['Gender'] != 'Other'], 
        x = 'AgeGroup', 
        y = 'Salary', 
        color = 'Gender',  
        template = "plotly_white",
        labels = {'AgeGroup': 'Age Group', 'Salary': 'Average Salary'},
        line_shape = 'spline',
        color_discrete_map = {'Male': male_colour, 'Female': female_colour},
        markers = True
    )

    fig_age.update_traces(
        fill='tozeroy',  
        fillcolor='rgba(115, 93, 165, 0.1)' # purple: 'rgba(115, 93, 165, 0.1)' green: 'rgba(106, 201, 58, 0.1)'
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
        f"""
        <style>
        .st-emotion-cache-6qob1r {{background-color: {sidebar_colour};}}
        .st-emotion-cache-1jicfl2 {{
            padding-top: 3rem;
            padding-bottom: 1rem;
            padding-left: 20px;
            padding-right: 20px;
                    }}
        .css-1d391kg {{
            padding-top: 3rem;
            padding-right: 1rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
        }}

        div.stButton > button:first-child {{
            background-color: #00cc00;
            color: white;
        }}

        div.stButton > button:first-child:hover {{
            border-color: #828282;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(f'<h1 style="color:{female_colour};">Machine Learning Model Predicting Employee Salary</h1>', unsafe_allow_html=True)

    
    def get_user_input():        
        education_level_list = ["Bachelor's Degree", "High School", "Master's Degree", "PhD"]
        department_list = ['Administration',
       'Finance', 'Human Resources',
       'Management', 'Operations', 'Other',
       'Product & Project Management',
       'Research & Strategy', 'Sales & Marketing',
       'Technology']
        department_list_options = [
       'Finance', 'Human Resources',
       'Management', 'Operations', 'Other',
       'Product & Project Management',
       'Research & Strategy', 'Sales & Marketing',
       'Technology']
        
        department = st.selectbox("Department", options = department_list_options, index = 0)
        age = st.number_input('Age', min_value=18)
        experience = st.number_input('Expereince',min_value=0)
        education_level = st.selectbox("Education Level", options = education_level_list, index = 0)

        department_dict = {f'Department_{dep}': int(dep == department) for dep in department_list}
        education_level_dict = {f'Education Level Updated_{edu}': int(edu == education_level) for edu in education_level_list}

        user_input = pd.DataFrame({
            'Age': [age],
            'Years of Experience': [experience],
            **education_level_dict,
            **department_dict,
        })

        st.text('')

        return user_input
    
    def transform_input_data(user_input):
        user_input_transformed = pd.get_dummies(user_input)
        return user_input_transformed
    
    def make_prediction(user_input):
        transformed_input = transform_input_data(user_input)
        model = joblib.load('random_forest_model.pkl') 
        prediction = model.predict(transformed_input)
        return prediction

    st.markdown('<p style="color:#ff0000;"><strong>Disclaimer:</strong> This model has been developed using a composite dataset obtained from Kaggle.com. Please note that the accuracy of the model has not been verified, and there is no supporting evidence to guarantee its predictive reliability. As such, all predictions generated by this model should be treated with caution and used accordingly. The full model can be found <strong><a href="https://github.com/lukecarroll101/SalaryDashboard/blob/main/EmployeeSalaryModel.ipynb" style="color:#ff0000;">here</a></strong>.</p>', unsafe_allow_html=True)
    st.text('Please fill in the details below to get the prediction.')

    user_input = get_user_input()

    prediction = make_prediction(user_input)
    prediction = prediction[0] * 1.55

    if 'clicked' not in st.session_state:
            st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    st.button('Submit', on_click=click_button)

    if st.session_state.clicked:
        # The message and nested widget will remain on the page
        st.markdown(f'<h3 style="color:{female_colour};">Salary Prediction: ${prediction:,.2f} </h3>', unsafe_allow_html=True)
        st.session_state.clicked = False



st.set_page_config(page_title="Employee Salary", page_icon="💼", layout="wide")


page_names_to_funcs = {
    "Dashboard": dashboard,
    "Machine Learning Model": ML_Deployment
}

demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()






# -----Offcuts------
#
# def add_line_breaks(text):
#             words = text.split(' ')
#             return '<br>'.join([' '.join(words[i:i+2]) for i in range(0, len(words), 2)])

# gender_race_salary['Race'] = gender_race_salary['Race'].apply(add_line_breaks)