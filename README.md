Cyber Shield AI – Intelligent Cybersecurity Threat Detection System
Project Overview

Cyber Shield AI is an intelligent cybersecurity analytics platform designed to detect and analyze multiple types of digital threats using machine learning and artificial intelligence.

The system integrates phishing detection, network intrusion detection, and insider threat analysis into a unified platform. It provides an interactive dashboard that enables security teams to monitor threats, analyze uploaded logs, and receive automated insights.

The platform is implemented using Python, Streamlit, and machine learning models to simulate a real-world cybersecurity monitoring environment.

Problem Statement

Modern organizations generate large volumes of security logs and network activity data. Manually identifying threats such as phishing emails, malicious network activity, or insider data breaches is challenging and time-consuming.

The objective of this project is to build an intelligent security system that can:

Detect phishing attacks from email data

Identify malicious network traffic

Detect potential insider threats

Provide automated threat analysis

Assist users with an AI-powered cybersecurity assistant

System Features
Security Dashboard

The platform provides a centralized dashboard showing:

Total devices monitored

Number of secure devices

Devices at risk

Active detected threats

This enables quick monitoring of overall system security.

Device Monitoring System

Users can:

Add and manage protected devices

View device status and operating system

Perform security scans

Monitor detected threats per device

Each device is categorized as:

Secure

Warning

Threat detected

File Scanner and Threat Analysis

The system allows users to upload security logs in CSV format.
The platform automatically detects the type of threat data and applies the appropriate machine learning model.

Supported analysis types include:

Phishing Detection

Detects malicious email messages

Uses NLP and machine learning models

Outputs phishing probability and classification

Network Intrusion Detection

Analyzes network traffic features

Identifies malicious activity

Predicts whether traffic represents an attack or normal behavior

Insider Threat Detection

Detects suspicious user behavior

Uses behavioral anomaly detection

Flags potential internal security risks

AI Security Assistant

Cyber Shield includes an AI-powered chatbot that assists users with cybersecurity questions.

The assistant can:

Explain detected threats

Suggest mitigation strategies

Provide cybersecurity recommendations

Assist with threat investigation

Automated Threat Alerts

When a threat is detected, the system generates alerts such as:

Suspicious email activity

Network anomalies

Potential insider threats

Alerts include a confidence score and suggested actions.

AI-Generated Security Insights

The platform can generate automated summaries of detected threats using a language model.

The system analyzes prediction results and produces insights such as:

Threat distribution

Key security risks

Recommended mitigation strategies

Technologies Used
Programming Languages

Python

Machine Learning

Scikit-learn

LightGBM

Random Forest

Natural Language Processing

Libraries

Pandas

NumPy

Joblib

Visualization & Interface

Streamlit

AI Integration

OpenAI API (optional)

How to Run the Project
Clone the repository
git clone https://github.com/YashasD11/Cyber-Shield-AI.git
Navigate to the project directory
cd cyber-shield-ai
Install required libraries
pip install -r requirements.txt
Run the application
streamlit run final_interface_app.py

The application will open in your browser.

Example Use Case

Login to the dashboard

Upload a security log dataset

The system automatically detects the dataset type

Machine learning models analyze the data

Threat predictions and alerts are generated

AI generates security insights

Applications

Cyber Shield AI can be used for:

Cybersecurity monitoring systems

Threat detection platforms

Network intrusion detection systems

Email phishing detection

Security analytics research
