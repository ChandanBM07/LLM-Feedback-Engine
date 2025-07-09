🤖 LLM-Product-Feedback-Dashboard
An interactive Streamlit-based dashboard that analyzes and visualizes customer reviews using advanced NLP techniques. Designed for product managers, UX researchers, and data analysts, this tool extracts sentiment, topics, and suggestions from app store reviews to support data-driven product development.

📌 Overview
The LLM-Product-Feedback-Dashboard is a powerful AI-driven tool built using Streamlit and transformer-based NLP models. It helps teams analyze large volumes of app store reviews to extract meaningful sentiment, topics, and product improvement suggestions — all from unstructured text.

Whether you're a product manager, UX researcher, data analyst, or part of a customer insights team, this dashboard simplifies the process of understanding what users are saying about your product.

🧾 Why This Tool?
Manually analyzing thousands of customer reviews is time-consuming and error-prone. This tool automates the process using state-of-the-art NLP models like:

distilbert-base-uncased-finetuned-sst-2-english for sentiment analysis

facebook/bart-large-mnli for zero-shot topic classification

🔍 You can instantly:

Understand how customers feel about your product.

Identify the most frequent topics (e.g., bugs, UI issues, features).

Detect recurring complaints or feature requests.

Make data-driven product decisions without relying on expensive SaaS tools.

🎯 Who Is It For?
✅ Product Managers – to prioritize features and improve satisfaction.

✅ UX Researchers – to discover pain points and usability issues.

✅ Data Analysts – to enrich reports with qualitative insights.

✅ Startups & Enterprises – to build customer-centric strategies.

✅ Students – learning NLP or building data visualization portfolios.

🚀 Features
✅ Sentiment Analysis using distilbert-base-uncased-finetuned-sst-2-english

✅ Zero-shot Topic Classification with facebook/bart-large-mnli

✅ Streamlit Dashboard for real-time exploration

✅ Filters by App Name, Category, Sentiment, and Keywords

✅ Interactive Visualizations: Word Clouds, Sentiment Charts

✅ Export Options: Download filtered data as .csv or .txt

✅ Feature Request Detection using regex-based keyword spotting

✅ Negative Review Suggestions for product improvement

✅ Fully local execution (no external API dependency)

🧠 How it Works
Preprocess reviews to clean placeholder or short reviews.

Classify sentiment using Hugging Face's DistilBERT model.

Classify topics using BART-based zero-shot classification.

Detect feature requests using simple NLP rules & regex.

Visualize insights in an intuitive Streamlit dashboard.

Enable filtering/exporting based on app, category, or sentiment.

🗂️ Project Structure
vbnet
Copy code
llm-feedback-dashboard/
├── app_dashboard.py                  # Streamlit app (main file)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
├── Procedure to run Streamlit.pdf    # Step-by-step run instructions
├── data/
│   ├── hybrid_sentiment_output.csv
│   ├── Apple_Store_Reviews.csv
│   ├── Play Store Data.csv
│   ├── combined_app_reviews.csv
│   ├── sentiment_labeled_app_reviews.csv
│   └── ...                           # Other processed datasets
