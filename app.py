# import getpass
# import re 
# import warnings
# import json
# import os
# import streamlit as st
# import textwrap
# # import streamlit_chat as message
# from langchain_community.document_loaders import CSVLoader, PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from utils import *

# # Set configuration
# load_dotenv()
# warnings.filterwarnings("ignore")
# gemini_api_key = os.getenv("GOOGLE_API_KEY")

# def main():
#     st.set_page_config(page_title='MedicalSpeak', page_icon='dockspeak_.png')
#     st.header("DocSpeak")
#     st.markdown("#### Chat with PDFs")
#     st.warning('Be respectful while asking questions')
#     user_question = st.text_input("Ask a Question from the PDF Files")
#     st.sidebar.markdown("# DocSpeak")
#     api_key = st.sidebar.text_input('Enter Gemini API key and Press Enter', type="password")
#     pdf_docs = st.sidebar.file_uploader("Upload your PDF Files.", accept_multiple_files=True)

#     if user_question:
#         user_input(user_question, api_key)

#     # with st.sidebar:
#     if st.sidebar.button("Submit & Process"):
#         if api_key:
#             with st.spinner("Embedding..."):
#                 raw_text = get_pdf_text(pdf_docs)

#                 ## Add image for chat
                
#                 text_chunks = get_text_chunks(raw_text)
#                 _ = get_vector_store(text_chunks, api_key)
#                 st.sidebar.success("Ready to Go!")
#         else:
#             st.sidebar.error('Please provide API key!')


# if __name__ == "__main__":
#     main()  

  






import getpass
import re 
import warnings
import json
import os
import streamlit as st
import textwrap
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from utils import *  # Import functions from utils.py

# Set configuration
load_dotenv()
warnings.filterwarnings("ignore")
gemini_api_key = os.getenv("GOOGLE_API_KEY")


def format_medical_info(medical_info):
    # Convert the dictionary items into a formatted string
    medical_context = ", ".join([f"{key}: {value}" for key, value in medical_info.items()])
    return medical_context

def combine_question_with_medical_info(user_question, medical_info):
    medical_context = format_medical_info(medical_info)
    combined_question = f"{user_question} [Medical Context: {medical_context}]"
    return combined_question




def main():
    st.set_page_config(page_title='MedicalSpeak', page_icon='dockspeak_.png')
    st.header("DocSpeak")
    st.markdown("#### Chat with PDFs")
    st.warning('Be respectful while asking questions')
    
    # Sidebar inputs
    user_question = st.text_input("Ask a Question from the PDF Files")
    st.sidebar.markdown("# DocSpeak")
    api_key = st.sidebar.text_input('Enter Gemini API key and Press Enter', type="password")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files.", accept_multiple_files=True)
    
    # Hydration Tracking
    st.sidebar.subheader("Hydration Tracker")
    daily_water_intake = st.sidebar.number_input("Enter your daily water intake (in liters)", min_value=0.0, step=0.1)

    # Stress Level Tracking
    st.sidebar.subheader("Stress Level Tracker")
    stress_level = st.sidebar.slider("Rate your stress level (1-10)", 1, 10)

    # Sleep Pattern Tracking
    st.sidebar.subheader("Sleep Pattern Tracker")
    sleep_hours = st.sidebar.slider("Enter hours of sleep", 0, 24, step=1)

    # Nutritional Diet Plan
    st.sidebar.subheader("Nutritional Diet Plan")
    diet_preference = st.sidebar.selectbox("Select your diet preference", ["Vegetarian", "Non-Vegetarian", "Vegan"])

    # Create medical_info dictionary outside the button click
    medical_info = {
        'daily_water_intake': daily_water_intake,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'diet_preference': diet_preference
    }

    # Combine the question with medical info if there's a user question
    combined_question = ""
    if user_question:
        combined_question = combine_question_with_medical_info(user_question, medical_info)

    # Submit & Process button
    if st.sidebar.button("Submit & Process"):
        if api_key:
            with st.spinner("Embedding..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                _ = get_vector_store(text_chunks, api_key)
                st.sidebar.success("Ready to Go!")
        else:
            st.sidebar.error('Please provide API key!')

    # Process the combined question
    if user_question and combined_question:  # Check both conditions

        st.markdown(combined_question)
        user_input(combined_question, api_key)
   

if __name__ == "__main__":
    main()


















# import streamlit as st
# from PyPDF2 import PdfReader

# def extract_pdf_text(pdf_file):
#     """Extract text from uploaded PDF file."""
#     pdf_reader = PdfReader(pdf_file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text() or ""
#     return text

# def analyze_health_factors(pdf_text, user_data):
#     """Analyze extracted PDF text along with user-provided health information."""
#     # Placeholder analysis logic; this can be expanded with more detailed NLP processing.
#     summary = f"Analyzing health factors for age {user_data['age']}, symptoms: {user_data['symptoms']}"
    
#     # Here you could implement specific analysis logic based on the PDF content.
    
#     # Example of potential health issues based on the analysis
#     health_issues = []
#     if "high blood pressure" in pdf_text.lower():
#         health_issues.append("High Blood Pressure")
#     if "anemia" in pdf_text.lower():
#         health_issues.append("Anemia")
    
#     return {"summary": summary, "health_issues": health_issues}

# # Title of the application
# st.title("Health Assistant")

# # Hydration Tracking Section
# st.header("Hydration Tracking")
# age = st.text_input("Enter your age:", key="age_input")
# weight = st.text_input("Enter your weight (kg):", key="weight_input")
# activity_level = st.text_input("Enter Activity Level (1-10):", key="activity_level_input")

# if st.button("Get Hydration Recommendation"):
#     # Predefined hydration recommendation logic
#     try:
#         age = int(age)
#         weight = float(weight)
#         activity_level = int(activity_level)

#         if age < 18:
#             recommendation = "Drink at least 1.5 liters of water daily."
#         elif age < 65:
#             recommendation = "Drink at least 2.0 liters of water daily."
#         else:
#             recommendation = "Drink at least 1.5 liters of water daily."
        
#         st.write(f"Recommended Hydration Intake: {recommendation}")

#     except ValueError:
#         st.error("Please enter valid numeric values.")

# # Nutrition Analysis Section
# st.header("Nutrition Analysis")
# diet_type = st.text_input("Enter Diet Type (Vegetarian/Vegan/Omnivore):", key="diet_type_input")
# caloric_intake = st.text_input("Enter Daily Caloric Intake:", key="caloric_intake_input")

# if st.button("Get Nutritional Suggestion"):
#     # Predefined nutritional suggestions based on diet type
#     try:
#         caloric_intake = float(caloric_intake)

#         if diet_type.lower() == "vegetarian":
#             suggestion = "Ensure you get enough protein from legumes and dairy."
#         elif diet_type.lower() == "vegan":
#             suggestion = "Include a variety of plant-based proteins and consider B12 supplementation."
#         else:
#             suggestion = "Focus on lean meats and include plenty of fruits and vegetables."
        
#         st.write(f"Nutritional Suggestion: {suggestion}")

#     except ValueError:
#         st.error("Please enter a valid caloric intake.")

# # Sleep Patterns Section
# st.header("Sleep Patterns")
# sleep_duration = st.text_input("Enter Average Sleep Duration (hours):", key="sleep_duration_input")
# sleep_quality = st.text_input("Rate Sleep Quality (1-10):", key="sleep_quality_input")

# if st.button("Get Sleep Improvement Advice"):
#     # Predefined sleep advice based on sleep quality
#     try:
#         sleep_duration = float(sleep_duration)
#         sleep_quality = int(sleep_quality)

#         if sleep_quality < 5:
#             advice = "Consider establishing a bedtime routine and reducing screen time before bed."
#         elif sleep_quality < 8:
#             advice = "Try to maintain a consistent sleep schedule and create a comfortable sleep environment."
#         else:
#             advice = "Great job! Keep maintaining healthy sleep habits."

#         st.write(f"Sleep Improvement Advice: {advice}")

#     except ValueError:
#         st.error("Please enter valid numeric values.")

# # Chronic Condition Diet Analysis Section
# st.header("Chronic Condition Diet Analysis")
# health_condition = st.text_input("Enter Health Condition (Diabetes/Hypertension/None):", key="health_condition_input")

# if st.button("Get Dietary Recommendation"):
#     # Predefined dietary recommendations based on health condition
#     if health_condition.lower() == "diabetes":
#         recommendation = "Focus on low glycemic index foods and monitor carbohydrate intake."
#     elif health_condition.lower() == "hypertension":
#         recommendation = "Reduce sodium intake and increase potassium-rich foods like bananas and spinach."
#     else:
#         recommendation = "Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins."

#     st.write(f"Dietary Recommendation: {recommendation}")

# # Pain Management Section
# st.header("Pain Management")
# pain_level = st.text_input("Rate Pain Level (1-10):", key="pain_level_input")
# activity_level_pm = st.text_input("Enter Activity Level (1-10):", key="activity_level_pm_input")

# if st.button("Get Pain Management Strategy"):
#     # Predefined pain management strategies based on pain level
#     try:
#         pain_level = int(pain_level)
#         activity_level_pm = int(activity_level_pm)

#         if pain_level >= 7:
#             strategy = "Consider consulting a healthcare provider for pain management options."
#         elif pain_level >= 4:
#             strategy = "Engage in light physical activity and consider over-the-counter pain relief."
#         else:
#             strategy = "Maintain regular physical activity and practice relaxation techniques."

#         st.write(f"Pain Management Strategy: {strategy}")

#     except ValueError:
#         st.error("Please enter valid numeric values.")

# # Blood Report Analysis Section
# st.header("Blood Report Analysis")
# uploaded_pdf_file = st.file_uploader("Upload your blood report PDF file", type=["pdf"])

# if uploaded_pdf_file is not None:
#     pdf_text = extract_pdf_text(uploaded_pdf_file)
    
#     user_data = {
#         "age": age,
#         "symptoms": health_condition,
#     }
    
#     analysis_results = analyze_health_factors(pdf_text, user_data)
    
#     if analysis_results:
#         st.write(analysis_results["summary"])
#         st.write(f"Potential Health Issues: {', '.join(analysis_results['health_issues'])}")









# import streamlit as st
# from PyPDF2 import PdfReader

# # Function to extract text from PDF file
# def extract_pdf_text(pdf_file):
#     """Extract text from uploaded PDF file."""
#     pdf_reader = PdfReader(pdf_file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text() or ""
#     return text

# # Function to analyze health factors from PDF and user input
# def analyze_health_factors(pdf_text, user_data):
#     """Analyze extracted PDF text along with user-provided health information."""
#     summary = f"Analyzing health factors for age {user_data['age']}, symptoms: {user_data['symptoms']}"
    
#     health_issues = []
#     if "high blood pressure" in pdf_text.lower():
#         health_issues.append("High Blood Pressure")
#     if "anemia" in pdf_text.lower():
#         health_issues.append("Anemia")
    
#     return {"summary": summary, "health_issues": health_issues}

# # Function to provide food recommendations based on age, weight, diet type, and health condition
# def get_food_recommendation(age, weight, diet_type, health_condition):
#     """Generate food recommendations based on age, weight, and health condition."""
#     recommended_food = []

#     # Caloric Intake Recommendations (basic example, adjust as needed)
#     caloric_intake = weight * 25  # Approximation: 25 calories per kg of weight
#     if age < 18:
#         caloric_intake += 200  # Teenagers need more calories
#     elif age > 50:
#         caloric_intake -= 200  # Older adults need fewer calories

#     recommended_food.append(f"Recommended Daily Caloric Intake: {caloric_intake} calories")

#     # Diet Type Specific Recommendations
#     if diet_type.lower() == "vegetarian":
#         recommended_food.append("Breakfast: Oatmeal with almond butter and chia seeds.")
#         recommended_food.append("Lunch: Lentil soup with whole grain bread and mixed salad.")
#         recommended_food.append("Dinner: Quinoa and vegetable stir-fry with tofu.")
#     elif diet_type.lower() == "vegan":
#         recommended_food.append("Breakfast: Smoothie with spinach, banana, and almond milk.")
#         recommended_food.append("Lunch: Chickpea salad with olive oil dressing.")
#         recommended_food.append("Dinner: Vegan lentil curry with brown rice.")
#     elif diet_type.lower() == "omnivore":
#         recommended_food.append("Breakfast: Scrambled eggs with avocado and whole wheat toast.")
#         recommended_food.append("Lunch: Grilled chicken with quinoa and steamed vegetables.")
#         recommended_food.append("Dinner: Salmon with roasted sweet potatoes and spinach.")

#     # Health Condition Specific Recommendations
#     if health_condition.lower() == "diabetes":
#         recommended_food.append("Avoid high-sugar foods. Include more low-GI foods like beans and whole grains.")
#     elif health_condition.lower() == "hypertension":
#         recommended_food.append("Reduce sodium intake. Include more potassium-rich foods like bananas, spinach, and potatoes.")
#     else:
#         recommended_food.append("Maintain a balanced diet with a variety of fruits, vegetables, lean proteins, and whole grains.")

#     return recommended_food

# # Streamlit Application

# # Title of the application
# st.title("Health Assistant")

# # Hydration Tracking Section
# st.header("Hydration Tracking")
# age = st.text_input("Enter your age:", key="age_input")
# weight = st.text_input("Enter your weight (kg):", key="weight_input")
# activity_level = st.text_input("Enter Activity Level (1-10):", key="activity_level_input")

# if st.button("Get Hydration Recommendation"):
#     try:
#         age = int(age)
#         weight = float(weight)
#         activity_level = int(activity_level)

#         if age < 18:
#             recommendation = "Drink at least 1.5 liters of water daily."
#         elif age < 65:
#             recommendation = "Drink at least 2.0 liters of water daily."
#         else:
#             recommendation = "Drink at least 1.5 liters of water daily."
        
#         st.write(f"Recommended Hydration Intake: {recommendation}")

#     except ValueError:
#         st.error("Please enter valid numeric values.")

# # Nutrition Analysis Section
# st.header("Nutrition Analysis")
# diet_type = st.text_input("Enter Diet Type (Vegetarian/Vegan/Omnivore):", key="diet_type_input")
# health_condition = st.text_input("Enter Health Condition (Diabetes/Hypertension/None):", key="health_condition_input")

# if st.button("Get Nutritional Suggestion"):
#     try:
#         # Get food recommendations based on age, weight, diet type, and health condition
#         food_recommendations = get_food_recommendation(int(age), float(weight), diet_type, health_condition)
        
#         st.write("\n".join(food_recommendations))

#     except ValueError:
#         st.error("Please enter valid numeric values.")

# # Pain Management Section
# st.header("Pain Management")
# pain_level = st.text_input("Rate Pain Level (1-10):", key="pain_level_input")
# activity_level_pm = st.text_input("Enter Activity Level (1-10):", key="activity_level_pm_input")

# if st.button("Get Pain Management Strategy"):
#     # Predefined pain management strategies based on pain level
#     try:
#         pain_level = int(pain_level)
#         activity_level_pm = int(activity_level_pm)

#         if pain_level >= 7:
#             strategy = "Consider consulting a healthcare provider for pain management options."
#         elif pain_level >= 4:
#             strategy = "Engage in light physical activity and consider over-the-counter pain relief."
#         else:
#             strategy = "Maintain regular physical activity and practice relaxation techniques."

#         st.write(f"Pain Management Strategy: {strategy}")

#     except ValueError:
#         st.error("Please enter valid numeric values.")

# # Sleep Patterns Section
# st.header("Sleep Patterns")
# sleep_duration = st.text_input("Enter Average Sleep Duration (hours):", key="sleep_duration_input")
# sleep_quality = st.text_input("Rate Sleep Quality (1-10):", key="sleep_quality_input")

# if st.button("Get Sleep Improvement Advice"):
#     # Predefined sleep advice based on sleep quality
#     try:
#         sleep_duration = float(sleep_duration)
#         sleep_quality = int(sleep_quality)

#         if sleep_quality < 5:
#             advice = "Consider establishing a bedtime routine and reducing screen time before bed."
#         elif sleep_quality < 8:
#             advice = "Try to maintain a consistent sleep schedule and create a comfortable sleep environment."
#         else:
#             advice = "Great job! Keep maintaining healthy sleep habits."

#         st.write(f"Sleep Improvement Advice: {advice}")

#     except ValueError:
#         st.error("Please enter valid numeric values.")

# # Stress Tracker Section
# st.header("Stress Tracker")
# stress_level = st.text_input("Rate Stress Level (1-10):", key="stress_level_input")

# if st.button("Get Stress Management Advice"):
#     # Predefined stress management strategies based on stress level
#     try:
#         stress_level = int(stress_level)

#         if stress_level >= 7:
#             strategy = "Engage in stress-reducing activities like meditation, yoga, or deep breathing exercises."
#         elif stress_level >= 4:
#             strategy = "Consider physical activities such as walking or stretching to relieve stress."
#         else:
#             strategy = "Keep up the good work in managing stress. Stay balanced and relax when needed."

#         st.write(f"Stress Management Strategy: {strategy}")

#     except ValueError:
#         st.error("Please enter valid numeric values.")

# # Blood Report Analysis Section
# st.header("Blood Report Analysis")
# uploaded_pdf_file = st.file_uploader("Upload your blood report PDF file", type=["pdf"])

# if uploaded_pdf_file is not None:
#     pdf_text = extract_pdf_text(uploaded_pdf_file)
    
#     user_data = {
#         "age": age,
#         "symptoms": health_condition,
#     }
    
#     analysis_results = analyze_health_factors(pdf_text, user_data)
    
#     if analysis_results:
#         st.write(analysis_results["summary"])
#         st.write(f"Potential Health Issues: {', '.join(analysis_results['health_issues'])}")
        
#         # Food recommendations based on health analysis
#         food_recommendations = get_food_recommendation(int(age), float(weight), diet_type, health_condition)
#         st.write("\n".join(food_recommendations))
