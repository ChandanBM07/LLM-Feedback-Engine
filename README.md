# LLM-Product-Feedback-Dashboard
An interactive Streamlit-based dashboard that analyzes and visualizes customer reviews using advanced NLP techniques. Designed to support product managers and analysts in understanding user sentiment, prioritizing features, and identifying improvement opportunities from app reviews.

## 🚀 Features

- ✅ **Sentiment Analysis** using `distilbert-base-uncased-finetuned-sst-2-english`
- ✅ **Zero-shot Topic Classification** with `facebook/bart-large-mnli`
- ✅ **Streamlit Dashboard** for real-time exploration
- ✅ **Filters by App Name, Category, Sentiment, and Keywords**
- ✅ **Interactive Visualizations**: Word Clouds, Sentiment Charts
- ✅ **Export Options**: Download filtered data as `.csv` or `.txt`
- ✅ **Feature Request Detection** using regex-based keyword spotting
- ✅ **Negative Review Suggestions** for product improvement
- ✅ **Fully local execution** (no external API dependency)

## 🧠 How it Works

1. **Preprocess reviews** to clean placeholder or short reviews.
2. **Classify sentiment** using Hugging Face's DistilBERT model.
3. **Classify topics** using BART-based zero-shot classification.
4. **Detect feature requests** using simple NLP rules & regex.
5. **Visualize insights** in an intuitive Streamlit dashboard.
6. **Enable filtering/exporting** based on app/category/sentiment.

---

## 📁 Project Structure
llm-feedback-dashboard/
├── app_dashboard.py # Streamlit app (main file)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Ignore unnecessary files
├── images/ # Dashboard screenshots
│ ├── dashboard.png
│ ├── sentiment_chart.png
│ └── wordcloud.png
└── data/
└── hybrid_sentiment_output.csv # Sample dataset
