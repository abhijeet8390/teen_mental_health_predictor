from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('depression_model.pkl', 'rb'))
le    = pickle.load(open('label_encoder.pkl', 'rb'))
oe    = pickle.load(open('ordinal_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get form data
    age                       = int(request.form['age'])
    gender                    = int(request.form['gender'])
    daily_social_media_hours  = float(request.form['daily_social_media_hours'])
    sleep_hours               = float(request.form['sleep_hours'])
    screen_time_before_sleep  = float(request.form['screen_time_before_sleep'])
    academic_performance      = float(request.form['academic_performance'])
    physical_activity         = float(request.form['physical_activity'])
    social_interaction_level  = float(request.form['social_interaction_level'])
    stress_level              = int(request.form['stress_level'])
    anxiety_level             = int(request.form['anxiety_level'])
    addiction_level           = int(request.form['addiction_level'])
    platform                  = request.form['platform_usage']

    # Handle platform one hot encoding
    platform_instagram = 1 if platform == 'instagram' else 0
    platform_tiktok    = 1 if platform == 'tiktok' else 0

    # Create dataframe
    input_data = pd.DataFrame([[
        age, gender, daily_social_media_hours,
        sleep_hours, screen_time_before_sleep,
        academic_performance, physical_activity,
        social_interaction_level, stress_level,
        anxiety_level, addiction_level,
        platform_instagram, platform_tiktok
    ]], columns=[
        'age', 'gender', 'daily_social_media_hours',
        'sleep_hours', 'screen_time_before_sleep',
        'academic_performance', 'physical_activity',
        'social_interaction_level', 'stress_level',
        'anxiety_level', 'addiction_level',
        'platform_usage_Instagram', 'platform_usage_TikTok'
    ])

    # Predict
    prediction = model.predict(input_data)[0]

    # Result message
    if prediction == 1:
        result = "⚠️ High Risk of Depression"
    else:
        result = "✅ Low Risk of Depression"

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)