from flask import Flask, request, render_template
import pickle
import numpy as np

# Define the updated feature set used for the model
updated_feature_set = [
    'Smoking', 'Alcohol', 'RedMeat', 'Pollution_Yes', 'AutoimmuneDisorder_Yes', 
    'DiabetesHistory_Yes', 'Exercise', 'FruitsVeggies', 'SleepHours', 'MentalHealth'
]

# Load the trained model
with open('rf_final_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    # Render the home page with the input form
    return render_template('landing_page.html')

@app.route('/survey')
def survey():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    # Initialize an empty list for input features
    input_features = []

    for feature in updated_feature_set:
        # Retrieve the form input
        input_value = request.form.get(feature)

        # Check if the input value is not None and is a valid number
        try:
            # Convert the input to float and append to the list
            input_features.append(float(input_value))
        except (TypeError, ValueError):
            # Handle the case where input is None or not a valid number
            # You can return an error message or set a default value
            return f"Invalid input for {feature}. Please enter a valid number."
    # Calculate risk score based on input features
    risk_score = calculate_risk_score(input_features)

    # Add the calculated risk score to the features for model prediction
    input_features_with_score = input_features 

    # Model prediction
    prediction = model.predict([np.array(input_features_with_score)])[0]

    # Determine risk level based on prediction
    #risk_level = "Risk" if prediction == 1 else " Healthy"
    # Determine recommendations based on risk score
    if risk_score > 7:
        recommendation = ("<ul><li><strong>High Risk Alert!!</strong></li></ul>"
                          "<ul><li>Your risk score is high. We strongly recommend scheduling a medical check-up for further assurance.</li></ul>")
        risk_level = "high Risk"
    elif risk_score > 0:
        recommendation = ("<ul><li><strong>Healthy But better to consider following</strong></li></ul>"
                        "<ul><li>Consider modifying your diet to include more green leafy vegetables.</li>"
                          "<li>Engage in regular physical activities.</li>"
                          "<li>Focus on quitting unhealthy habits such as smoking or excessive alcohol consumption.</li></ul>")
        risk_level = "Moderate Risk should be Cautious"
    else:
        recommendation = "Congratulations on maintaining a healthy lifestyle! Keep up the good work."
        risk_level = "Healthy enough"


    return render_template('results.html', risk_level=risk_level,risk_score = risk_score,recommendation=recommendation)

def calculate_risk_score(features):
    """
    Calculate the risk score based on input features.
    """
    # Assigning weights to each feature
    weights = {
        'Smoking': 3, 'Alcohol': 3, 'RedMeat': 2, 'Pollution_Yes': 2, 
        'AutoimmuneDisorder_Yes': 2, 'DiabetesHistory_Yes': 2, 'Exercise': -1, 
        'FruitsVeggies': -1, 'SleepHours': -1, 'MentalHealth': -1
    }

    # Calculate weighted sum of the features
    risk_score = sum(features[i] * weights[updated_feature_set[i]] for i in range(len(features)))
    
    return risk_score

if __name__ == "__main__":
    app.run(debug=True)
