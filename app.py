from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


xgb_model = joblib.load(r'C:\Users\SThanuj\Desktop\EL\phase3_model.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index')
def index():
    
    return render_template('index.html')

@app.route('/about_project')
def about_project():
    
    return render_template('about_project.html')

@app.route('/moisture', methods=['GET', 'POST'])
def moisture():
    prediction = None  # Initialize prediction variable
    
    if request.method == 'POST':
        input_data = request.form.get('id1')
        
        try:
            # Convert input string to a numpy array
            input_array = np.array([list(map(float, input_data.split(',')))]).reshape(1, -1)  # Added reshape for proper array format
            
            # Predict moisture levels
            predicted_time = xgb_model.predict(input_array)
            prediction = f"{predicted_time[0]}"  # Store the prediction
        except Exception as e:
            prediction = f"Error: {str(e)}"  # If an error occurs, show error message
    
    return render_template('moisture.html', prediction=prediction)


@app.route('/faqs')
def faqs():
    return render_template('faqs.html')

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')

if __name__ == '__main__':
    app.run(debug=True)


        


