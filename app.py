#peoject 2
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from scipy.spatial import distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import shap
import matplotlib.pyplot as plt

def main():
	'''Set main() function. Includes sidebar navigation and respective routing.'''
	st.sidebar.title("Menu")
	app_mode = st.sidebar.selectbox( "Choose an Action", [
		"Welcome page",
        "Instruction",
		"Start Evaluation",

	])

	# nav
	if   app_mode == "Welcome page": show_about()
	elif app_mode == "Start Evaluation": explore_classified()
	elif app_mode == "Instruction":data_dictionary()

def show_about():
	''' Home page '''
	st.image("https://www.arubanetworks.com/wp-content/uploads/verisk-2-cs.jpg",
             use_column_width=True)
	st.title('Welcome to Doctor’s Helper Disease Detection System!')
	st.markdown("Last updated on Nov 1 2020") 
	st.subheader("Ya Liu")
	st.write("With this tool, you could quickly detect a patient's risk after you input patient's information. The system would provide following insights:")
	st.markdown("1. whether the disease is predicted to be present at this patient ")
	st.markdown("2. what metrics you should pay attention to")
	st.markdown("3. past similar cases in hospital's database with diagnostic result")
    
def data_dictionary():
    st.title('What information do I need to collect during a pre-surgical screening?')
    st.write("Here is a table for important metrics. You may refer to this table to understand variable meaning.")
    st.image("https://github.com/yaliu0703/yaliu0703.github.io/blob/master/images/data%20dict%20pic.png?raw=true",use_column_width=True)
    st.write("Generally speaking, patients meeting any of following criteria should be prioritized for an appointment:")
    st.markdown("1. Patients with high Blood Chemestry I/II")
    st.markdown("2. Old people or people with high BMI")
    st.markdown("3. People with high # pregnancies/Genetic Predeposition Factor")
    st.markdown("4. Patients from TX/CA")
    st.image("https://github.com/yaliu0703/yaliu0703.github.io/blob/master/images/global%20interpretation.png?raw=true",use_column_width=True)

##################################################################################
#                                                                                #
# Start Evaluation                                                               #
# ::: Allow the user to pick one row from dataset and evaluate                   #
# ::: and show interpretation                                                    #
#                                                                                #
##################################################################################

def explore_classified():
	# Text on Evaluation page 
	st.title("Doctor's Helper Disease Risk Estimation System")
	st.write('''
		Step 1. Input patient information at the side menu.
		Step 2. Click the button "Read risk analytics report" to get prediction result and interpretation.
	''')

	# Step 1:User chooses one row record 
	demo = st.radio('Choose a patient profile',("profile A","profile B"))  # Input the index number
	# Step 2:Get user input
	if demo == "profile A": newdata = get_input(int(3))#bad
	elif demo == "profile B":newdata = get_input(int(37))#good
   
	# Step 3: when user checks, run model
	if st.button('Read risk analytics report'):
		run_model(newdata)

def get_input(index):
    
    values = X_train.iloc[index]  # Input the value from dataset

    # Create input variables for evaluation please use these variables for evaluation
    
    NumberofPregnancies = st.sidebar.slider('NumberofPregnancies', 0, 17, int(values[0]),1)
    BloodChemestryI = st.sidebar.text_input('Blood Chemestry I:', values[1])
    BloodChemestryII = st.sidebar.text_input('Blood Chemestry II:', values[2])
    BloodPressure = st.sidebar.text_input('Blood Pressure:', values[3])
    SkinThickness = st.sidebar.slider('Skin Thickness',0.0, 100.0, float(values[4]))
    BMI = st.sidebar.slider('BMI:', 15.0, 67.0, float(values[5]))
    GeneticPredispositionFactor = st.sidebar.slider('Genetic Predisposition Factor:', -1.0, 1.0, float(values[6]))
    Age = st.sidebar.text_input('Age:', values[7],4)
    AirQualityIndex = st.sidebar.slider('Air Quality Index:', 1.0, 100.0, float(values[8]))
    State = st.sidebar.selectbox('Patient living in', df.State.unique().tolist())

    
    newdata = pd.DataFrame()
    newdata = newdata.append({'# Pregnancies':NumberofPregnancies,
            'Blood Chemestry I':BloodChemestryI,
            'Blood Chemestry II':BloodChemestryII,
            'Blood Pressure':BloodPressure,
            'Skin Thickness':SkinThickness,
            'BMI':BMI,
            'Genetic Predisposition Factor':GeneticPredispositionFactor,
            "Age": Age,
            'Air Quality Index':AirQualityIndex,
            'State': State},ignore_index=True)
    return newdata
        
def run_model(newdata):
    st.write(newdata)
    y_pred = rf_model.predict(newdata)
    if y_pred[0] > 0:
        st.text("This patient is identified as one of high risk group. Please contact the patient to schedule an appointment for him.")
    else:
        st.text("This patient is less likely to affect the disease. If there are patients with a more urgent situation, we may prioritize other risky patients and contact this patient later.")
    st.subheader("Detailed information:")
    choosen_instance = pd.DataFrame(rf_model.named_steps["preprocessor"].transform(newdata), columns=all_features)
    shap_values = shap.TreeExplainer(rf_model.named_steps["model"]).shap_values(choosen_instance)
    shap.initjs()
    shap.force_plot(shap.TreeExplainer(rf_model.named_steps["model"]).expected_value[1], shap_values[1], choosen_instance,matplotlib=True,show=False
                    ,figsize=(16,5))
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    plt.clf()
    find_similar_record(newdata)

def find_similar_record(newdata):
    newdata = newdata[num_features].apply(pd.to_numeric)
    newdata_normalized = (newdata - df.iloc[:,0:9].mean()) / df.iloc[:,0:9].std() 
    # Find the distance between lebron james and everyone else.
    euclidean_distances = df_normalized.apply(lambda row: distance.euclidean(row, newdata_normalized), axis=1)
    # Create a new dataframe with distances.
    distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
    distance_frame.sort_values("dist", inplace=True) 
    # Find the most similar record to new data 
    second_smallest = distance_frame.iloc[1:4]["idx"]
    most_similar_to_input = df.iloc[second_smallest]
    st.subheader("Here are historical records similar to your input for your reference:")
    st.write(most_similar_to_input)    

    
##################################################################################
#                                                                                #
# Execute                                                                        #
#                                                                                #
##################################################################################


if __name__ == "__main__":
	
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    df = pd.read_csv("df.csv")
    num_features = ["# Pregnancies", "Blood Chemestry I", "Blood Chemestry II", "Blood Pressure", "Skin Thickness", 
                "BMI", "Genetic Predisposition Factor","Age", "Air Quality Index"]
    cat_features = ["State"]
   
    model = RandomForestClassifier(class_weight='balanced', max_depth=10,
                                        min_samples_split=10, n_jobs=-1)
    preprocessor = ColumnTransformer([("numerical", "passthrough", num_features), 
                                  ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                   cat_features)])
    rf_model = Pipeline([("preprocessor", preprocessor), 
                     ("model", RandomForestClassifier(class_weight="balanced", n_estimators=100, n_jobs=-1))])
    rf_model.fit(X_train, y_train)
    preprocessor = rf_model.named_steps["preprocessor"]
    ohe_categories = preprocessor.named_transformers_["categorical"].categories_
    new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]
    all_features = num_features + new_ohe_features
    # Normalize all of the numeric columns
    df_normalized = (df.iloc[:,0:9] - df.iloc[:,0:9].mean()) / df.iloc[:,0:9].std()
    
   
	# execute
    main()
  
    
    
    
    