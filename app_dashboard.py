# ========================================================================
# AI REVIEW ANALYZER - FIXED & COMPLETE VERSION
# All bugs fixed, charts working, robust error handling
# ========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import pipeline
import torch
import requests
import json
from datetime import datetime
from io import BytesIO
import re
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# PAGE CONFIGURATION
# ========================================================================
st.set_page_config(
    page_title="AI Review Analyzer Pro",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 AI Review Analyzer Pro</p>', unsafe_allow_html=True)

# ========================================================================
# SESSION STATE INITIALIZATION
# ========================================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# ========================================================================
# SIDEBAR: DATA SOURCE SELECTION
# ========================================================================
st.sidebar.title("📊 Data Source")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Choose how to load data:",
    ["📁 Upload CSV File", "🔌 Connect to API", "🎲 Load Sample Data"]
)

df = None

# ========================================================================
# OPTION 1: UPLOAD CSV FILE
# ========================================================================
if data_source == "📁 Upload CSV File":
    st.sidebar.markdown("### Upload Your CSV File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload any CSV file with review data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Show preview
            with st.sidebar.expander("👀 Preview Data"):
                st.dataframe(df.head(3))
        
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            st.stop()

# ========================================================================
# OPTION 2: API INTEGRATION
# ========================================================================
elif data_source == "🔌 Connect to API":
    st.sidebar.markdown("### API Configuration")
    
    api_type = st.sidebar.selectbox(
        "Select API Source:",
        ["Custom REST API", "CSV URL", "Google Play Store API", "Apple App Store API"]
    )
    
    if api_type == "Custom REST API":
        api_url = st.sidebar.text_input(
            "API Endpoint URL:",
            placeholder="https://api.example.com/reviews"
        )
        
        use_auth = st.sidebar.checkbox("Requires Authentication?")
        api_key = ""
        
        if use_auth:
            auth_type = st.sidebar.selectbox("Auth Type:", ["Bearer Token", "API Key"])
            api_key = st.sidebar.text_input("Token/Key:", type="password")
        
        if st.sidebar.button("🚀 Fetch Data from API"):
            if api_url:
                with st.spinner("Fetching data from API..."):
                    try:
                        headers = {}
                        if use_auth and api_key:
                            if auth_type == "Bearer Token":
                                headers["Authorization"] = f"Bearer {api_key}"
                            else:
                                headers["X-API-Key"] = api_key
                        
                        response = requests.get(api_url, headers=headers, timeout=30)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            if isinstance(data, list):
                                df = pd.DataFrame(data)
                            elif isinstance(data, dict):
                                for key in ['data', 'results', 'reviews', 'items']:
                                    if key in data and isinstance(data[key], list):
                                        df = pd.DataFrame(data[key])
                                        break
                                if df is None:
                                    df = pd.DataFrame([data])
                            
                            st.sidebar.success(f"✅ Fetched {len(df):,} records from API")
                        else:
                            st.sidebar.error(f"❌ API Error: Status {response.status_code}")
                    
                    except Exception as e:
                        st.sidebar.error(f"❌ Error: {str(e)}")
            else:
                st.sidebar.warning("⚠️ Please enter an API URL")
    
    elif api_type == "CSV URL":
        csv_url = st.sidebar.text_input(
            "CSV File URL:",
            placeholder="https://example.com/data.csv"
        )
        
        if st.sidebar.button("📥 Load CSV from URL"):
            if csv_url:
                try:
                    df = pd.read_csv(csv_url)
                    st.sidebar.success(f"✅ Loaded {len(df):,} rows from URL")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading CSV: {str(e)}")
    
    else:
        st.sidebar.info(f"📱 {api_type} integration coming soon!")

# ========================================================================
# OPTION 3: SAMPLE DATA
# ========================================================================
elif data_source == "🎲 Load Sample Data":
    if st.sidebar.button("Load Sample Reviews"):
        df = pd.DataFrame({
            'Review': [
                "This app is amazing! Love all the new features and smooth performance.",
                "Terrible experience. App crashes every time I try to login.",
                "Good app overall but really needs a dark mode option.",
                "Best productivity app I've ever used! Highly recommend to everyone.",
                "Too many bugs. Please fix the sync issues ASAP.",
                "Decent app but the UI could be more intuitive.",
                "Complete waste of money. Doesn't work as advertised.",
                "Great customer support team! They solved my issue quickly.",
                "App is okay but missing some key features I need for work.",
                "Perfect! Does exactly what I need it to do.",
                "Love this app! The new update made it even better.",
                "Keeps freezing on my device. Very frustrating.",
                "Would be 5 stars if it had offline mode.",
                "Excellent design and easy to use interface.",
                "Not worth the subscription price at all."
            ],
            'App_Name': ['TaskPro', 'TaskPro', 'NoteMaster', 'TaskPro', 'NoteMaster', 
                        'NoteMaster', 'TaskPro', 'NoteMaster', 'TaskPro', 'NoteMaster',
                        'TaskPro', 'NoteMaster', 'TaskPro', 'NoteMaster', 'TaskPro'],
            'Rating': [5, 1, 4, 5, 2, 3, 1, 5, 3, 5, 5, 2, 4, 5, 1]
        })
        st.sidebar.success(f"✅ Loaded {len(df)} sample reviews")

# ========================================================================
# SMART COLUMN MAPPING
# ========================================================================
if df is not None and not df.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗂️ Column Mapping")
    st.sidebar.caption("Map your CSV columns to the required fields")
    
    columns = df.columns.tolist()
    
    # Smart auto-detection
    def find_column(keywords):
        for col in columns:
            if any(keyword in col.lower() for keyword in keywords):
                return col
        return columns[0]
    
    detected_review = find_column(['review', 'text', 'comment', 'feedback', 'content', 'description'])
    detected_app = find_column(['app', 'name', 'product', 'title'])
    detected_rating = find_column(['rating', 'score', 'stars'])
    
    # User column selection
    review_col = st.sidebar.selectbox(
        "Review Text Column:",
        columns,
        index=columns.index(detected_review)
    )
    
    app_col = st.sidebar.selectbox(
        "App Name Column:",
        columns,
        index=columns.index(detected_app)
    )
    
    rating_col = st.sidebar.selectbox(
        "Rating Column (optional):",
        ["None"] + columns,
        index=columns.index(detected_rating) + 1 if detected_rating in columns else 0
    )
    
    # Apply column mapping
    df_mapped = df.copy()
    df_mapped['Review'] = df[review_col].astype(str)
    df_mapped['App_Name'] = df[app_col].astype(str)
    
    if rating_col != "None":
        df_mapped['Rating'] = pd.to_numeric(df[rating_col], errors='coerce')
    
    # Clean data
    df_mapped = df_mapped.dropna(subset=['Review'])
    df_mapped = df_mapped[df_mapped['Review'].str.len() >= 10]
    df_mapped = df_mapped.drop_duplicates(subset=['Review'])
    df_mapped = df_mapped.reset_index(drop=True)
    
    st.session_state.df = df_mapped
    df = df_mapped
    
    st.sidebar.info(f"📊 Ready: {len(df):,} reviews")

# ========================================================================
# MAIN ANALYSIS SECTION
# ========================================================================
if st.session_state.df is not None:
    df = st.session_state.df
    
    st.markdown("## 🔬 AI Analysis Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_sentiment = st.checkbox("💭 Sentiment Analysis", value=True)
    with col2:
        run_topics = st.checkbox("🏷️ Topic Classification", value=True)
    with col3:
        run_features = st.checkbox("⭐ Feature Extraction", value=True)
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        batch_size = st.slider("Processing Batch Size:", 8, 64, 32)
        
        custom_topics = st.text_input(
            "Custom Topics (comma-separated):",
            value="Bugs, Features, UI/UX, Performance, Pricing, Support, Login Issues"
        )
        topics_list = [t.strip() for t in custom_topics.split(',')]
    
    # ========================================================================
    # RUN ANALYSIS BUTTON
    # ========================================================================
    if st.button("🚀 Analyze Reviews", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # Load models
            device = 0 if torch.cuda.is_available() else -1
            
            # SENTIMENT ANALYSIS
            if run_sentiment:
                status.text("🔄 Loading sentiment model...")
                sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=device
                )
                
                status.text("🔄 Analyzing sentiment...")
                reviews = df['Review'].fillna("").tolist()
                all_sentiments = []
                all_confidences = []
                
                for i in range(0, len(reviews), batch_size):
                    batch = reviews[i:i+batch_size]
                    batch = [str(r)[:512] for r in batch]
                    
                    try:
                        results = sentiment_model(batch)
                        for r in results:
                            label = r['label'].upper()
                            # Normalize labels
                            if 'POSITIVE' in label or 'POS' in label or 'LABEL_2' in label:
                                all_sentiments.append('POSITIVE')
                            elif 'NEGATIVE' in label or 'NEG' in label or 'LABEL_0' in label:
                                all_sentiments.append('NEGATIVE')
                            else:
                                all_sentiments.append('NEUTRAL')
                            all_confidences.append(r['score'])
                    except:
                        all_sentiments.extend(['NEUTRAL'] * len(batch))
                        all_confidences.extend([0.5] * len(batch))
                    
                    progress_bar.progress(min((i + batch_size) / len(reviews) * 0.4, 0.4))
                
                df['Sentiment'] = all_sentiments
                df['Confidence'] = all_confidences
                
                status.text(f"✅ Sentiment analysis complete! Found {(df['Sentiment']=='POSITIVE').sum()} positive reviews")
            
            # TOPIC CLASSIFICATION
            if run_topics:
                status.text("🔄 Loading topic classifier...")
                topic_model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=device
                )
                
                status.text("🔄 Classifying topics...")
                all_topics = []
                all_topic_conf = []
                
                for i, review in enumerate(df['Review'].tolist()):
                    try:
                        result = topic_model(str(review)[:512], topics_list, multi_label=False)
                        all_topics.append(result['labels'][0])
                        all_topic_conf.append(result['scores'][0])
                    except:
                        all_topics.append('Unknown')
                        all_topic_conf.append(0.0)
                    
                    if i % 10 == 0:
                        progress_bar.progress(min(0.4 + (i / len(df)) * 0.5, 0.9))
                
                df['Topic'] = all_topics
                df['Topic_Confidence'] = all_topic_conf
                
                status.text(f"✅ Topic classification complete!")
            
            # FEATURE REQUEST EXTRACTION
            if run_features:
                status.text("🔄 Extracting features...")
                
                patterns = [
                    r'\b(wish|want|need|hope)\b.*\b(feature|option|function)',
                    r'\bplease\s+(add|include|give|provide)',
                    r'\b(should|could|would)\s+(have|add|include)',
                    r'\b(missing|lack)\b',
                    r'\bfeature\s+request\b',
                ]
                combined = '|'.join(patterns)
                
                df['Is_Feature_Request'] = df['Review'].str.contains(
                    combined, case=False, regex=True, na=False
                )
            
            progress_bar.progress(1.0)
            status.empty()
            progress_bar.empty()
            
            st.session_state.analyzed = True
            st.session_state.df = df
            
            st.success("✅ Analysis Complete!")
            st.balloons()
            
            # Show summary
            st.markdown("### 📋 Quick Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                if 'Sentiment' in df.columns:
                    st.write("**Sentiment Breakdown:**")
                    sent_counts = df['Sentiment'].value_counts()
                    for sent, count in sent_counts.items():
                        pct = (count / len(df)) * 100
                        st.write(f"- {sent}: {count} ({pct:.1f}%)")
            
            with summary_col2:
                if 'Topic' in df.columns:
                    st.write("**Top 3 Topics:**")
                    top_topics = df['Topic'].value_counts().head(3)
                    for topic, count in top_topics.items():
                        st.write(f"- {topic}: {count}")
            
            with summary_col3:
                if 'Is_Feature_Request' in df.columns:
                    feat_count = df['Is_Feature_Request'].sum()
                    feat_pct = (feat_count / len(df)) * 100
                    st.write(f"**Feature Requests:**")
                    st.write(f"- Found: {feat_count}")
                    st.write(f"- Percentage: {feat_pct:.1f}%")
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    # ========================================================================
    # RESULTS DASHBOARD
    # ========================================================================
    if st.session_state.analyzed:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", f"{len(df):,}")
        
        with col2:
            if 'Rating' in df.columns:
                avg_rating = df['Rating'].mean()
                if pd.notna(avg_rating):
                    st.metric("Avg Rating", f"{avg_rating:.2f}/5")
                else:
                    st.metric("Avg Rating", "N/A")
            else:
                st.metric("Avg Rating", "No ratings")
        
        with col3:
            if 'Sentiment' in df.columns:
                total = len(df)
                pos_count = (df['Sentiment'] == 'POSITIVE').sum()
                pos_pct = (pos_count / total * 100) if total > 0 else 0
                st.metric("Positive %", f"{pos_pct:.1f}%")
            else:
                st.metric("Positive %", "Not analyzed")
        
        with col4:
            if 'Confidence' in df.columns:
                st.metric("Avg Confidence", f"{df['Confidence'].mean():.1%}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📋 Data", "☁️ Word Clouds", "📥 Export"])
        
        with tab1:
            if 'Sentiment' in df.columns:
                st.markdown("### Sentiment & Topic Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Sentiment Distribution")
                    sentiment_counts = df['Sentiment'].value_counts()
                    
                    if len(sentiment_counts) > 0:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
                        
                        wedges, texts, autotexts = ax.pie(
                            sentiment_counts.values, 
                            labels=sentiment_counts.index,
                            autopct='%1.1f%%',
                            colors=colors[:len(sentiment_counts)],
                            startangle=90
                        )
                        
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontsize(12)
                            autotext.set_weight('bold')
                        
                        ax.set_title('Overall Sentiment', fontsize=14, fontweight='bold')
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    if 'Topic' in df.columns:
                        st.markdown("#### Top Topics")
                        topic_counts = df['Topic'].value_counts().head(8)
                        
                        if len(topic_counts) > 0:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.barh(topic_counts.index, topic_counts.values, color='skyblue')
                            ax.set_xlabel('Count', fontsize=12)
                            ax.set_title('Most Discussed Topics', fontsize=14, fontweight='bold')
                            ax.grid(axis='x', alpha=0.3)
                            st.pyplot(fig)
                            plt.close()
                
                # Sentiment by App
                if 'App_Name' in df.columns and df['App_Name'].nunique() > 1:
                    st.markdown("### Sentiment by App")
                    
                    app_sentiment = pd.crosstab(df['App_Name'], df['Sentiment'], normalize='index') * 100
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    app_sentiment.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#95a5a6'], width=0.7)
                    ax.set_title('Sentiment Distribution by App', fontsize=14, fontweight='bold')
                    ax.set_xlabel('App', fontsize=12)
                    ax.set_ylabel('Percentage (%)', fontsize=12)
                    ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1))
                    ax.grid(axis='y', alpha=0.3)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                    plt.close()
        
        with tab2:
            st.markdown("### Review Data")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'Sentiment' in df.columns:
                    filter_sentiment = st.multiselect(
                        "Filter by Sentiment:",
                        options=df['Sentiment'].unique().tolist(),
                        default=df['Sentiment'].unique().tolist()
                    )
            
            with col2:
                if 'Topic' in df.columns:
                    all_topics = sorted(df['Topic'].unique().tolist())
                    filter_topic = st.multiselect(
                        "Filter by Topic:",
                        options=all_topics,
                        default=all_topics
                    )
            
            with col3:
                if 'App_Name' in df.columns:
                    all_apps = sorted(df['App_Name'].unique().tolist())
                    filter_app = st.multiselect(
                        "Filter by App:",
                        options=all_apps,
                        default=all_apps
                    )
            
            # Apply filters
            filtered = df.copy()
            if 'Sentiment' in df.columns and filter_sentiment:
                filtered = filtered[filtered['Sentiment'].isin(filter_sentiment)]
            if 'Topic' in df.columns and filter_topic:
                filtered = filtered[filtered['Topic'].isin(filter_topic)]
            if 'App_Name' in df.columns and filter_app:
                filtered = filtered[filtered['App_Name'].isin(filter_app)]
            
            # Search
            search_term = st.text_input("🔍 Search reviews:", placeholder="Enter keywords...")
            if search_term:
                filtered = filtered[filtered['Review'].str.contains(search_term, case=False, na=False)]
            
            st.dataframe(
                filtered[['Review', 'App_Name', 'Sentiment', 'Topic', 'Confidence']].head(100),
                use_container_width=True,
                height=400
            )
            
            st.caption(f"Showing {min(len(filtered), 100):,} of {len(filtered):,} filtered reviews (from {len(df):,} total)")
        
        with tab3:
            if 'Sentiment' in df.columns:
                st.markdown("### Word Clouds")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Positive Reviews")
                    pos_reviews = df[df['Sentiment'] == 'POSITIVE']
                    if len(pos_reviews) > 0:
                        pos_text = ' '.join(pos_reviews['Review'].astype(str))
                        if len(pos_text) > 50:
                            wc = WordCloud(
                                width=600, 
                                height=400, 
                                background_color='white',
                                colormap='Greens',
                                max_words=100
                            ).generate(pos_text)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.info("Not enough positive reviews to generate word cloud")
                    else:
                        st.info("No positive reviews found")
                
                with col2:
                    st.markdown("#### Negative Reviews")
                    neg_reviews = df[df['Sentiment'] == 'NEGATIVE']
                    if len(neg_reviews) > 0:
                        neg_text = ' '.join(neg_reviews['Review'].astype(str))
                        if len(neg_text) > 50:
                            wc = WordCloud(
                                width=600, 
                                height=400, 
                                background_color='white',
                                colormap='Reds',
                                max_words=100
                            ).generate(neg_text)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.info("Not enough negative reviews to generate word cloud")
                    else:
                        st.info("No negative reviews found")
        
        with tab4:
            st.markdown("### Export Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox("Format:", ["CSV", "Excel", "JSON"])
            
            with col2:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = st.text_input("Filename:", value=f"analysis_{timestamp}")
            
            if export_format == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    f"{filename}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            elif export_format == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Analysis')
                    
                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Total Reviews', 'Positive %', 'Negative %', 'Neutral %', 'Avg Confidence'],
                        'Value': [
                            len(df),
                            f"{(df['Sentiment']=='POSITIVE').sum()/len(df)*100:.1f}%" if 'Sentiment' in df.columns else 'N/A',
                            f"{(df['Sentiment']=='NEGATIVE').sum()/len(df)*100:.1f}%" if 'Sentiment' in df.columns else 'N/A',
                            f"{(df['Sentiment']=='NEUTRAL').sum()/len(df)*100:.1f}%" if 'Sentiment' in df.columns else 'N/A',
                            f"{df['Confidence'].mean():.1%}" if 'Confidence' in df.columns else 'N/A'
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, index=False, sheet_name='Summary')
                
                st.download_button(
                    "⬇️ Download Excel",
                    buffer.getvalue(),
                    f"{filename}.xlsx",
                    "application/vnd.ms-excel",
                    use_container_width=True
                )
            
            elif export_format == "JSON":
                json_str = df.to_json(orient='records', indent=2)
                st.download_button(
                    "⬇️ Download JSON",
                    json_str,
                    f"{filename}.json",
                    "application/json",
                    use_container_width=True
                )

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome to AI Review Analyzer Pro
    
    **Features:**
    - 📁 **Upload any CSV file** with review data
    - 🔌 **Connect to APIs** for real-time data
    - 💭 **AI-powered sentiment analysis** using state-of-the-art models
    - 🏷️ **Automatic topic classification** with zero-shot learning
    - ⭐ **Feature request detection** using NLP patterns
    - 📊 **Interactive visualizations** with charts and word clouds
    - 📥 **Export results** in CSV, Excel, or JSON formats
    
    #### Getting Started:
    1. **Choose a data source** from the sidebar (Upload CSV, API, or Sample Data)
    2. **Map your columns** - The app will auto-detect review and rating columns
    3. **Click "Analyze Reviews"** to run AI analysis
    4. **Explore insights** in the interactive dashboard!
    
    ---
    
    **💡 Pro Tip:** Start with sample data to see how it works, then upload your own CSV file!
    """)
    
    # Example CSV format
    with st.expander("📄 Expected CSV Format"):
        st.markdown("""
        Your CSV should have at least these columns (names can vary):
        
        | Review/Text/Comment | App Name | Rating (optional) |
        |---------------------|----------|-------------------|
        | "Great app!"        | MyApp    | 5                 |
        | "Needs improvement" | MyApp    | 3                 |
        
        **Supported column name variations:**
        - **Review column:** review, text, comment, feedback, content, description
        - **App column:** app, name, product, title
        - **Rating column:** rating, score, stars (1-5 scale)
        """)

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;color:#666;padding:1rem;'>
        Made with ❤️ using Streamlit & Hugging Face Transformers | 
        <a href='https://github.com' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
