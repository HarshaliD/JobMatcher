import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import google.generativeai as genai
import io

# Configure API key for the generative model
genai.configure(api_key="AIzaSyBp-RzlAZAjsaZAwFX4CyFFRFVsADUxjKc")

# Load the pre-trained KNN model
with open("knn.pkl", "rb") as model_file:
    knn = pickle.load(model_file)

# Load dataset
@st.cache
def load_data():
    return pd.read_csv("C:/Users/lenovo/Documents/GIT/ML End to End/JobMatcher/cs_students.csv")

data = load_data()

# Map skill levels to numerical values
skill_mapping = {'Strong': 2, 'Average': 1, 'Weak': 0}
data['Python'] = data['Python'].map(skill_mapping)
data['SQL'] = data['SQL'].map(skill_mapping)
data['Java'] = data['Java'].map(skill_mapping)

# One-hot encode the 'Interested Domain' column
data_encoded = pd.get_dummies(data, columns=['Interested Domain'], drop_first=True)

# Automatically encode 'Future Career'
label_encoder = LabelEncoder()
data_encoded['Career'] = label_encoder.fit_transform(data_encoded['Future Career'])

# Drop unnecessary columns
data_processed = data_encoded.drop(columns=['Student ID', 'Name', 'Gender', 'Future Career', 'Major', 'Projects'])

# Split features and target
X = data_processed.drop('Career', axis=1)
y = data_processed['Career']

# Input function for new student
def input_new_student():
    gpa = st.number_input("Enter GPA:", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
    python_skill = st.selectbox("Python skill:", ["Strong", "Average", "Weak"])
    sql_skill = st.selectbox("SQL skill:", ["Strong", "Average", "Weak"])
    java_skill = st.selectbox("Java skill:", ["Strong", "Average", "Weak"])
    domain = st.selectbox("Interested domain:", ["Software Development", "Data Science", "AI/ML", "Other"])

    python_skill_num = skill_mapping[python_skill]
    sql_skill_num = skill_mapping[sql_skill]
    java_skill_num = skill_mapping[java_skill]

    new_student_profile = pd.DataFrame({
        'GPA': [gpa],
        'Python': [python_skill_num],
        'SQL': [sql_skill_num],
        'Java': [java_skill_num]
    })

    domain_encoded = pd.get_dummies([domain], columns=['Interested Domain'], drop_first=True)
    new_student_profile = pd.concat([new_student_profile, domain_encoded], axis=1)

    missing_cols = set(data_processed.columns) - set(new_student_profile.columns)
    for col in missing_cols:
        new_student_profile[col] = 0

    new_student_profile = new_student_profile[X.columns]
    return new_student_profile

# Function to recommend careers using KNN
def recommend_careers(student_profile, knn_model, top_n=3):
    distances, indices = knn_model.kneighbors(student_profile)
    recommended_careers = y.iloc[indices[0]].values[:top_n]
    recommended_career_names = label_encoder.inverse_transform(recommended_careers)
    return recommended_career_names

# Generating the report
# Function to generate the PDF with Unicode font support
def generate_pdf(skills_info, careers_list, response_text):
    # Replace unsupported characters (like en dash) with supported ones
    skills_info = skills_info.replace("–", "-")
    careers_list = careers_list.replace("–", "-")
    response_text = response_text.replace("–", "-")

    # You can add more replacements here for other unsupported characters if needed
    # For example:
    # response_text = response_text.replace("©", "(c)")  # Replace copyright symbol

    pdf = FPDF()
    pdf.add_page()

    # Use a default built-in font that does not require external font files
    pdf.set_font('Arial', '', 12)  # You can replace 'Arial' with other built-in fonts (e.g., 'Times', 'Courier')

    pdf.cell(200, 10, txt="Career Recommendation Report", ln=True, align='C')

    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt="Skills Information:", ln=True)
    pdf.multi_cell(0, 10, skills_info)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Recommended Careers:", ln=True)
    pdf.multi_cell(0, 10, careers_list)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Career Preparation Guide:", ln=True)
    pdf.multi_cell(0, 10, response_text)

    # Save the PDF to a byte stream (in-memory)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)  # Rewind the file pointer

    return pdf_output

# Streamlit app interface
def main():
    st.title("Career Recommendation System")

    # Input section for a new student profile
    new_student_profile = input_new_student()

    # Submit button to generate career recommendation
    if st.button("Generate Career Report"):
        # Recommend careers for the new student
        recommended_careers_for_new_student = recommend_careers(new_student_profile, knn)

        # Prepare the input for the generative model
        careers_list = ", ".join(recommended_careers_for_new_student)

        # Extract skills and input details for context
        python_skill = new_student_profile.iloc[0]['Python']
        sql_skill = new_student_profile.iloc[0]['SQL']
        java_skill = new_student_profile.iloc[0]['Java']
        gpa = new_student_profile.iloc[0]['GPA']

        skills_info = (
            f"The student has a GPA of {gpa}, with the following skill levels:\n"
            f"- Python: {'Strong' if python_skill == 2 else 'Average' if python_skill == 1 else 'Weak'}\n"
            f"- SQL: {'Strong' if sql_skill == 2 else 'Average' if sql_skill == 1 else 'Weak'}\n"
            f"- Java: {'Strong' if java_skill == 2 else 'Average' if java_skill == 1 else 'Weak'}\n"
        )

        prompt = (
            f"{skills_info}\n"
            f"Based on their profile, the system has recommended the following careers: {careers_list}. "
            f"Please provide a detailed, step-by-step guide on how the student can prepare and apply for these careers. "
            f"Include required skills, certifications, job search strategies, and networking tips."
        )

        # Generate content with the generative model
        flash = genai.GenerativeModel('gemini-1.5-flash')
        response = flash.generate_content(prompt)
        response_text = response.text

        # Display the recommended careers and preparation guide
        st.subheader("Recommended Careers")
        st.write(careers_list)

        st.subheader("Career Preparation Guide")
        st.write(response_text)

        # Provide the PDF download link
        pdf_report = generate_pdf(skills_info, careers_list, response_text)

        # Allow the user to download the generated PDF
        st.download_button(
            label="Download PDF Report",
            data=pdf_report,
            file_name="career_recommendation_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
