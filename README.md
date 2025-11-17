<H1>🚗 AI Vehicle Maintenance Predictor
Predict vehicle maintenance needs using Machine Learning & Flask</H1>

This project analyzes real vehicle sensor data such as mileage, engine temperature, RPM, oil pressure, and fuel efficiency to determine whether a vehicle requires maintenance soon.
It uses an XGBoost classifier with scaled inputs and provides results via a modern dashboard UI built with HTML, Bootstrap and Flask.

✨ Features
Feature	Description
🔧 Predictive Maintenance	AI model predicts if maintenance is required
📊 ML Model	XGBoost classifier trained with real dataset
🌐 Web App	Flask backend + Bootstrap UI dashboard
📁 Clean Structure	Ready for deployment and scaling
📦 Stored Model	Uses .pkl for model and scaler
💻 Deployment Ready	Works on Render, Railway, Vercel (backend setup)
🧠 Machine Learning Workflow
Step	Details
Data preprocessing	Scaling numeric inputs using StandardScaler
Model training	XGBoost classifier
Evaluation	Accuracy score + Classification report
Saving	model.pkl + scaler.pkl stored for runtime prediction
