from pathlib import Path

import streamlit as st
from PIL import Image

#--Path Settings
curr_dir=Path(__file__).parent if "__file__" in locals() else Path.cwd()

css_file=curr_dir/"styles"/"main.css"
resume_file=curr_dir/"assets"/"cv.pdf"
profile_pic=curr_dir/"assets"/"profile-pic.png"


#--General Settings
PAGE_TITLE ="Digital CV | Vishnu Palanisamy"
PAGE_ICON=":WAVE:"
NAME="Vishnu Palanisamy"
DESCRIPTION="""
Research Analyst @ Amazon, Assisting with Data Analysis & Decision making-Maintaining smooth process flow by collabrating with various Teams.
"""
EMAIL="vishnuansel@gmail.com"
SOCIAL_MEDIA ={
    "LinkedIn":"https://www.linkedin.com/in/vishnu-palanisamy-52619a100/",
    "GitHub":"https://github.com/palavish",
}
CONTACT="+91-9003434371"

PROJECTS ={
    "Chennai Hosue Price Prediction":"https://github.com/palavish/Projects/blob/main/House-price-prediction-chennai.ipynb",
    "Tweet Sentiment Analysis":"https://github.com/palavish/NLP-/blob/main/Twitter%20Sentiment%20Analysis%20-NLP.ipynb",
    "Tweets Data Analysis":"https://github.com/palavish/NLP/blob/main/NLP-Tweets%20(Company%20Assessment).ipynb",
    "Breast Cancer Prediction system with NN":"https://github.com/palavish/Machine-Learning/blob/main/Breast%20Cancer%20Prediction%20system%20with%20NN.ipynb",
    "Credit Card Fraud Detection":"https://github.com/palavish/Machine-Learning/blob/main/Credit%20Card%20Fraud%20Detection.ipynb",
    "Customer Segmentation":"https://github.com/palavish/Machine-Learning/blob/main/Customer%20Segmentation.ipynb",
    "Diabetes Prediction":"https://github.com/palavish/Machine-Learning/blob/main/Diabetes%20Prediction.ipynb",
    "Heart Disease Prediction":"https://github.com/palavish/Machine-Learning/blob/main/Heart%20Disease%20Prediction.ipynb",
    "Object detection OpenCV":"https://github.com/palavish/Machine-Learning/blob/main/Object%20detection%20OpenCV.ipynb",
}

st.set_page_config(page_title=PAGE_TITLE,page_icon=PAGE_ICON)


#--Load CSS,PDF & Profile pic
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)
with open(resume_file,"rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)

#---Head Section
col1,col2 =st.columns(2,gap="small")
with col1:
    st.image(profile_pic,width=300)

with col2:
    st.title(NAME)
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte, 
        file_name=resume_file.name,
        mime="application/octet-stream",
    )
    st.write("📧",EMAIL)
    st.write("📱",CONTACT)

#-- Social Meida
st.write("#")
cols=st.columns(len(SOCIAL_MEDIA))
for index,(platform,link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")

#--Experience & Qualifications
st.write("#")
st.subheader("Experience & Qualifications")
st.write("""
        - ③ - 3 Years Experience @ Amazon(APR2021-present**)
        - Demonstrated commitment to continuous learning and professional development.
        - Received Stellar Award twice(2022 & 2023)in Amazon for my work.
    """)
#--Skills
st.write("#")
st.subheader("Hard Skills")
st.write("""
        ◢Technical Skills:
        - ✭Hands on Experience and Knowledge in Python ,Excel & MySql
        - ✭Experience with data manipulation and analysis libraries, including pandas, NumPy, and SciPy,NLP.
        - ✭Knowledge of machine learning frameworks and libraries, such as scikit-learn, TensorFlow, Keras
        - ✭Familiarity with statistical methods and techniques for hypothesis testing, regression analysis, and predictive modeling.
        - ✭Learning DJANGO,FLASK,KIVY,OpenCV 
             
        ◢Data Handling and Analysis:
        - ✭Experience with data cleaning, preprocessing, and feature engineering techniques to prepare datasets for analysis.
        - ✭Strong understanding of data visualization principles and proficiency with visualization tools like Matplotlib, Seaborn, or Plotly.
        - ✭Ability to interpret and communicate insights from data analysis to stakeholders using clear and concise visualizations and reports.
            
        ◢Machine Learning and Data Modeling:
        - ✭Knowledge of various machine learning algorithms, including supervised and unsupervised learning methods, classification, regression, clustering, and dimensionality reduction techniques.
        - ✭Experience with model evaluation and validation techniques, cross-validation, hyperparameter tuning, and model selection.
        - ✭Familiarity with deep learning concepts and neural network architectures for tasks such as image recognition, natural language processing, or time series analysis.
            
        ◢Research and Analytical Skills:
        - ✭Strong analytical and problem-solving skills, with the ability to approach complex problems logically and systematically.
        - ✭Experience conducting research, literature reviews, and experimental design for data-driven projects.
        - ✭Ability to critically evaluate data sources, methodologies, and results to ensure accuracy and reliability.
            
        ◢Communication and Collaboration:
        - ✭Effective communication skills, both verbal and written, for presenting findings, explaining technical concepts, and collaborating with multidisciplinary teams.
        - ✭Experience working in collaborative research environments, contributing to team projects, and sharing knowledge with colleagues.
        - ✭Ability to translate technical concepts and insights into layman's terms for non-technical stakeholders.
    """ )

#--Projects and Accomplishments
st.write("#")
st.subheader("Projects")
st.write("---")
for project,link in PROJECTS.items():
    st.write(f"[{project}]({link})")
