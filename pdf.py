from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Zomato Restaurant Analysis And Predict Their Ratings', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Title Page
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Zomato Restaurant Analysis And Predict Their Ratings', 0, 1, 'C')
pdf.cell(0, 10, '12.06.2024', 0, 1, 'C')
pdf.cell(0, 10, 'Lavish Gangwani', 0, 1, 'C')
pdf.ln(20)

# Document Control
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Document Control', 0, 1, 'L')
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, 'Date       | Version | Description                             | Author', 0, 1, 'L')
pdf.cell(0, 10, '12/06/2024 | 1.0     | Abstract, Introduction, General Description | Lavish Gangwani', 0, 1, 'L')
pdf.cell(0, 10, '14/06/2024 | 1.1     | Design Detail, API Deployment             | Lavish Gangwani', 0, 1, 'L')
pdf.cell(0, 10, '17/06/2024 | 1.2     | Final Revision                            | Lavish Gangwani', 0, 1, 'L')
pdf.ln(10)

# Abstract
pdf.chapter_title('Abstract')
abstract = (
    "The objective of this project is to perform extensive Exploratory Data Analysis (EDA) on the Zomato dataset and build a "
    "predictive Machine Learning model to estimate restaurant ratings based on specific features. This will help Zomato restaurants "
    "anticipate their ratings and take strategic actions to enhance customer satisfaction and overall performance. The approach "
    "includes data exploration, data cleaning, feature engineering, model building, hyperparameter tuning, and model testing. "
    "The Random Forest Regressor achieved the highest predictive accuracy, providing a robust framework for predicting restaurant ratings."
)
pdf.chapter_body(abstract)

# Introduction
pdf.chapter_title('Introduction')
introduction = (
    "1.1 Purpose of High-Level Design Document\n"
    "The purpose of this High-Level Design (HLD) Document is to add necessary detail to the project description to represent a suitable model for coding. "
    "It helps detect contradictions prior to coding and can be used as a reference manual for module interactions at a high level.\n\n"
    "1.2 Scope\n"
    "The HLD documentation presents the structure of the system, such as database architecture, application architecture (layers), application flow (Navigation), "
    "and technology architecture. It uses non-technical to mildly-technical terms understandable to the system administrators. It includes detailed descriptions of user interfaces, "
    "hardware and software interfaces, performance requirements, design features, and non-functional attributes like security, reliability, maintainability, portability, "
    "application compatibility, and resource utilization."
)
pdf.chapter_body(introduction)

# General Description
pdf.chapter_title('General Description')
general_description = (
    "2.1 Product Perspective & Problem Statement\n"
    "The goal of this project is to analyze restaurants in Bangalore and predict their ratings using a dataset containing information on 70-80 thousand restaurants. "
    "The problem involves calculating the ratings of individual restaurants based on provided data.\n\n"
    "2.2 Tools Used\n"
    "- Python: Programming language for model building and API development.\n"
    "- Jupyter Notebooks: For exploratory data analysis and model development.\n"
    "- Pandas, NumPy: Libraries for data manipulation and numerical operations.\n"
    "- Scikit-Learn: Machine learning library for model building.\n"
    "- FastAPI: Framework for building APIs.\n"
    "- SQLite: Database for storing processed data.\n"
    "- Google Colab: For collaborative code development and testing."
)
pdf.chapter_body(general_description)

# Design Detail
pdf.chapter_title('Design Detail')
design_detail = (
    "3.1 Functional Architecture\n"
    "The functional architecture includes the following components:\n"
    "- Data Ingestion: Collect data from Zomato's dataset.\n"
    "- Data Cleaning: Handle missing values, correct inconsistencies, and prepare the dataset for modeling.\n"
    "- Feature Engineering: Transform raw data into meaningful features.\n"
    "- Model Building: Train various machine learning models (Random Forest, Linear Regression, etc.).\n"
    "- Model Evaluation: Evaluate models using appropriate metrics to determine the best fit.\n"
    "- API Development: Develop an API using FastAPI to serve the model predictions.\n"
    "- Frontend Interface: Simple HTML form for inputting restaurant data and displaying predictions.\n\n"
    "3.2 Optimization\n"
    "Optimization strategies include:\n"
    "- Data Strategy: Minimize the number of fields and records, optimize extracts to speed up future queries.\n"
    "- Filter Optimization: Limit the number of filters and use include filters, continuous date filters, Boolean or numeric filters.\n"
    "- Calculation Optimization: Perform calculations in the database, reduce nested calculations, use MIN or MAX instead of AVG, make groups with calculations, use Booleans or numeric calculations."
)
pdf.chapter_body(design_detail)

# Key Performance Indicators (KPIs)
pdf.chapter_title('Key Performance Indicators (KPIs)')
kpis = (
    "- Restaurant Rating Summary: Displaying a summary of restaurant ratings and their relationship with different metrics.\n"
    "- Online Table Bookings: Percentage of people booking tables online or offline.\n"
    "- Location Metrics: Location and neighborhood of restaurants.\n"
    "- Online Orders: Whether restaurants accept online orders.\n"
    "- Popular Dishes: Most liked dishes of the restaurants.\n"
    "- Cuisines: Types of cuisines offered by the restaurants."
)
pdf.chapter_body(kpis)

# Deployment
pdf.chapter_title('Deployment')
deployment = (
    "FastAPI for API Deployment: Using FastAPI to build and deploy APIs for model predictions. FastAPI offers high performance, rapid development speed, fewer bugs, intuitive design, and ease of use. "
    "It reduces code duplication and supports efficient feature development.\n\n"
    "Deployment Steps:\n"
    "1. Setup Environment: Configure the Python environment and install required libraries.\n"
    "2. Develop API: Implement endpoints for predictions using FastAPI.\n"
    "3. Testing: Test the API endpoints to ensure correct functionality.\n"
    "4. Deployment: Deploy the API to a cloud service (e.g., AWS, Google Cloud) for scalability and availability.\n"
    "5. Monitor and Maintain: Monitor the API for performance and maintain it for any necessary updates or bug fixes."
)
pdf.chapter_body(deployment)

# Save the PDF
output_path = "/GeekDS/Zomato_Restaurant_Analysis_LLD.pdf"
pdf.output(output_path)

output_path
