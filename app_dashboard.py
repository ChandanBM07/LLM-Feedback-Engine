import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# Set page configuration
st.set_page_config(page_title="LLM Product Feedback Dashboard", page_icon="📱", layout="wide")

# Load data with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"D:\project-LLM\Data\hybrid_sentiment_output.csv")
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please check the file path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Data cleaning function
def clean_data(df):
    """Clean and preprocess the data"""
    df_clean = df.copy()
    
    # Clean placeholder or invalid reviews
    if "Review" in df_clean.columns:
        # Remove null reviews first
        df_clean = df_clean.dropna(subset=['Review'])
        
        # Convert to string and clean
        df_clean['Review'] = df_clean['Review'].astype(str)
        
        # Remove placeholder reviews
        placeholder_keywords = ["no review available", "no reviews", "no review", "noreview", "nan", "none"]
        for keyword in placeholder_keywords:
            df_clean = df_clean[~df_clean["Review"].str.lower().str.contains(keyword, na=False)]
        
        # Remove very short reviews (less than 10 characters)
        df_clean = df_clean[df_clean["Review"].str.len() > 10]
    
    # Enhanced Category cleaning
    if "Category" in df_clean.columns:
        # Remove rows where Category is NaN or empty
        df_clean = df_clean.dropna(subset=['Category'])
        
        # Convert to string and clean
        df_clean['Category'] = df_clean['Category'].astype(str)
        
        # Remove rows with 'nan', 'none', or empty string categories
        df_clean = df_clean[~df_clean['Category'].str.lower().isin(['nan', 'none', '', 'null'])]
        
        # Clean and standardize category names
        df_clean["Category"] = (df_clean["Category"]
                               .str.strip()  # Remove leading/trailing spaces
                               .str.lower()  # Convert to lowercase
                               .str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters
                               .str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with single space
                               .str.strip())  # Remove spaces again after cleaning
        
        # Remove any remaining empty categories
        df_clean = df_clean[df_clean["Category"] != '']
        df_clean = df_clean[df_clean["Category"].str.len() > 0]
    
    # Clean App_Name
    if "App_Name" in df_clean.columns:
        df_clean = df_clean.dropna(subset=['App_Name'])
        df_clean['App_Name'] = df_clean['App_Name'].astype(str).str.strip()
        df_clean = df_clean[df_clean['App_Name'] != '']
    
    # Clean Sentiment_Label
    if "Sentiment_Label" in df_clean.columns:
        df_clean = df_clean.dropna(subset=['Sentiment_Label'])
        df_clean['Sentiment_Label'] = df_clean['Sentiment_Label'].astype(str).str.strip().str.upper()
        # Standardize sentiment labels
        sentiment_mapping = {
            'POSITIVE': 'POSITIVE',
            'NEGATIVE': 'NEGATIVE',
            'NEUTRAL': 'NEUTRAL',
            'POS': 'POSITIVE',
            'NEG': 'NEGATIVE',
            'NEU': 'NEUTRAL'
        }
        df_clean['Sentiment_Label'] = df_clean['Sentiment_Label'].map(sentiment_mapping).fillna(df_clean['Sentiment_Label'])
    
    return df_clean

# Clean the data
df = clean_data(df)

# Display data info
st.sidebar.markdown(f"**Total Records**: {len(df):,}")
if not df.empty:
    st.sidebar.markdown(f"**Data Columns**: {', '.join(df.columns)}")

# Sidebar filters
st.sidebar.header("📊 Filter Options")

# App filter
if "App_Name" in df.columns and not df.empty:
    app_list = sorted(df["App_Name"].unique())
    st.sidebar.markdown(f"**Available Apps**: {len(app_list)}")
    
    # Option to select all apps or specific ones
    select_all_apps = st.sidebar.checkbox("Select All Apps", value=True)
    if select_all_apps:
        selected_apps = app_list
    else:
        selected_apps = st.sidebar.multiselect("Select App(s)", app_list, default=app_list[:5])
else:
    selected_apps = []
    st.sidebar.warning("⚠️ No 'App_Name' column found")

# Sentiment filter
if "Sentiment_Label" in df.columns and not df.empty:
    sentiments = sorted(df["Sentiment_Label"].unique())
    st.sidebar.markdown(f"**Available Sentiments**: {', '.join(sentiments)}")
    
    select_all_sentiments = st.sidebar.checkbox("Select All Sentiments", value=True)
    if select_all_sentiments:
        selected_sentiments = sentiments
    else:
        selected_sentiments = st.sidebar.multiselect("Select Sentiment(s)", sentiments, default=sentiments)
else:
    selected_sentiments = []
    st.sidebar.warning("⚠️ No 'Sentiment_Label' column found")

# Enhanced Category filter
if "Category" in df.columns and not df.empty:
    # Get all unique categories after cleaning
    categories = sorted(df["Category"].unique())
    
    st.sidebar.markdown(f"**📂 Available Categories**: {len(categories)}")
    
    # Show sample categories
    sample_categories = categories[:10]
    st.sidebar.markdown(f"*Sample*: {', '.join(sample_categories)}{'...' if len(categories) > 10 else ''}")
    
    # Option to select all categories or specific ones
    select_all_categories = st.sidebar.checkbox("Select All Categories", value=True)
    
    if select_all_categories:
        selected_categories = categories
    else:
        # Show categories in a more user-friendly way
        selected_categories = st.sidebar.multiselect(
            "Select Category(s)", 
            categories,
            default=[],
            help=f"Choose from {len(categories)} available categories"
        )
        
        # If no categories selected manually, show all
        if not selected_categories:
            selected_categories = categories
            st.sidebar.info("ℹ️ Showing all categories since none selected")
else:
    selected_categories = []
    st.sidebar.warning("⚠️ No 'Category' column found in data")

# Keyword search
search_term = st.sidebar.text_input("🔍 Search in Reviews", "", help="Search for specific words in reviews")

# Apply filters
if not df.empty:
    filtered_df = df.copy()
    
    # Apply app filter
    if selected_apps and "App_Name" in df.columns:
        filtered_df = filtered_df[filtered_df["App_Name"].isin(selected_apps)]
    
    # Apply sentiment filter
    if selected_sentiments and "Sentiment_Label" in df.columns:
        filtered_df = filtered_df[filtered_df["Sentiment_Label"].isin(selected_sentiments)]
    
    # Apply category filter
    if selected_categories and "Category" in df.columns:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
    
    # Apply keyword search
    if search_term and "Review" in df.columns:
        filtered_df = filtered_df[filtered_df["Review"].str.contains(search_term, case=False, na=False)]
else:
    filtered_df = pd.DataFrame()

# Title and description
st.title("📱 LLM Product Feedback Dashboard")
st.markdown("Analyze app reviews with sentiment analysis and category insights")

# Display data summary with better formatting
if not filtered_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Reviews", f"{len(filtered_df):,}")
    with col2:
        app_count = len(filtered_df["App_Name"].unique()) if "App_Name" in filtered_df.columns else 0
        st.metric("📱 Apps", app_count)
    with col3:
        cat_count = len(filtered_df["Category"].unique()) if "Category" in filtered_df.columns else 0
        st.metric("📂 Categories", cat_count)
    with col4:
        sent_count = len(filtered_df["Sentiment_Label"].unique()) if "Sentiment_Label" in filtered_df.columns else 0
        st.metric("😊 Sentiments", sent_count)
else:
    st.warning("⚠️ No data available with current filters")

# Visualization selection
if not filtered_df.empty:
    st.markdown("---")
    view_option = st.radio(
        "📈 Select Visualization", 
        ["Sentiment Breakdown", "Category Distribution", "Word Cloud", "App Performance"],
        horizontal=True
    )

    # 1. Sentiment Breakdown
    if view_option == "Sentiment Breakdown":
        st.subheader("📊 Sentiment Breakdown by App")
        
        if "App_Name" in filtered_df.columns and "Sentiment_Label" in filtered_df.columns:
            # Get top apps by review count
            top_apps = filtered_df["App_Name"].value_counts().head(10).index.tolist()
            subset = filtered_df[filtered_df["App_Name"].isin(top_apps)]

            if not subset.empty:
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.countplot(data=subset, x="App_Name", hue="Sentiment_Label", palette="Set2", ax=ax)
                plt.title("Sentiment Breakdown (Top 10 Apps by Review Count)", fontsize=16, pad=20)
                plt.xlabel("App Name", fontsize=12)
                plt.ylabel("Number of Reviews", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title="Sentiment", title_fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show sentiment percentages
                sent_pct = subset["Sentiment_Label"].value_counts(normalize=True) * 100
                st.subheader("📈 Overall Sentiment Distribution")
                col1, col2, col3 = st.columns(3)
                for i, (sentiment, pct) in enumerate(sent_pct.items()):
                    with [col1, col2, col3][i % 3]:
                        st.metric(f"{sentiment}", f"{pct:.1f}%")
            else:
                st.info("No data available for sentiment breakdown.")

    # 2. Enhanced Category Distribution
    elif view_option == "Category Distribution":
        st.subheader("📂 Distribution by Category")
        
        if "Category" in filtered_df.columns:
            category_counts = filtered_df["Category"].value_counts()
            
            if len(category_counts) > 0:
                # Bar chart
                fig, ax = plt.subplots(figsize=(14, 8))
                top_categories = category_counts.head(15)
                bars = ax.bar(range(len(top_categories)), top_categories.values, color='skyblue', alpha=0.8)
                ax.set_title(f"Top {len(top_categories)} Categories by Review Count", fontsize=16, pad=20)
                ax.set_xlabel("Category", fontsize=12)
                ax.set_ylabel("Number of Reviews", fontsize=12)
                ax.set_xticks(range(len(top_categories)))
                ax.set_xticklabels(top_categories.index, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Pie chart for top categories
                st.subheader("🥧 Category Distribution (Top 10)")
                top_10_cats = category_counts.head(10)
                fig, ax = plt.subplots(figsize=(10, 8))
                wedges, texts, autotexts = ax.pie(top_10_cats.values, labels=top_10_cats.index, 
                                                 autopct='%1.1f%%', startangle=90)
                ax.set_title("Top 10 Categories Distribution", fontsize=16, pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Category breakdown table
                st.subheader("📋 Detailed Category Breakdown")
                category_df = category_counts.reset_index()
                category_df.columns = ['Category', 'Count']
                category_df['Percentage'] = (category_df['Count'] / category_df['Count'].sum() * 100).round(2)
                category_df['Rank'] = range(1, len(category_df) + 1)
                category_df = category_df[['Rank', 'Category', 'Count', 'Percentage']]
                
                st.dataframe(category_df, use_container_width=True)
            else:
                st.info("No categories found in filtered data.")
        else:
            st.info("No category data available.")

    # 3. Word Cloud
    elif view_option == "Word Cloud":
        st.subheader("☁️ Review Word Cloud")
        
        if "Review" in filtered_df.columns:
            all_text = " ".join(filtered_df["Review"].dropna().astype(str))
            if all_text.strip():
                try:
                    wordcloud = WordCloud(
                        width=1200, 
                        height=600, 
                        background_color='white',
                        max_words=100,
                        colormap='Set2'
                    ).generate(all_text)
                    
                    fig, ax = plt.subplots(figsize=(15, 8))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    ax.set_title("Most Common Words in Reviews", fontsize=16, pad=20)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
            else:
                st.info("No reviews available to generate word cloud.")

    # 4. App Performance Analysis
    elif view_option == "App Performance":
        st.subheader("📱 App Performance Analysis")
        
        if "App_Name" in filtered_df.columns and "Sentiment_Label" in filtered_df.columns:
            # Calculate sentiment scores for each app
            app_sentiment = filtered_df.groupby(['App_Name', 'Sentiment_Label']).size().unstack(fill_value=0)
            
            if not app_sentiment.empty:
                # Calculate positive ratio
                app_sentiment['Total'] = app_sentiment.sum(axis=1)
                if 'POSITIVE' in app_sentiment.columns:
                    app_sentiment['Positive_Ratio'] = (app_sentiment['POSITIVE'] / app_sentiment['Total'] * 100).round(2)
                else:
                    app_sentiment['Positive_Ratio'] = 0
                
                # Sort by positive ratio
                app_sentiment = app_sentiment.sort_values('Positive_Ratio', ascending=False)
                
                # Display top and bottom performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 Top Performers")
                    top_performers = app_sentiment.head(5)
                    for app, row in top_performers.iterrows():
                        st.metric(app, f"{row['Positive_Ratio']:.1f}%", f"{int(row['Total'])} reviews")
                
                with col2:
                    st.subheader("⚠️ Needs Improvement")
                    bottom_performers = app_sentiment.tail(5)
                    for app, row in bottom_performers.iterrows():
                        st.metric(app, f"{row['Positive_Ratio']:.1f}%", f"{int(row['Total'])} reviews")

# Suggestions from negative reviews
if not filtered_df.empty and "Sentiment_Label" in filtered_df.columns:
    negative_reviews = filtered_df[filtered_df["Sentiment_Label"] == "NEGATIVE"]
    if not negative_reviews.empty and "Review" in negative_reviews.columns:
        st.markdown("---")
        st.subheader("💡 Insights from Negative Feedback")
        st.info("Sample negative feedback that may help guide improvements:")
        
        # Get a sample of negative reviews
        sample_size = min(5, len(negative_reviews))
        if sample_size > 0:
            sample_reviews = negative_reviews["Review"].dropna().sample(sample_size, random_state=42)
            for i, review in enumerate(sample_reviews, 1):
                st.write(f"**{i}.** *{review[:200]}{'...' if len(review) > 200 else ''}*")
        
        st.caption("💭 Consider addressing common themes in negative feedback to improve user satisfaction.")

# Export Section
st.markdown("---")
st.subheader("⬇️ Export Filtered Data")

if not filtered_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("Choose export format:", ["CSV", "Excel", "JSON"])
        filename = st.text_input("Enter filename (without extension):", value="filtered_reviews")
    
    with col2:
        st.markdown("**Export Summary:**")
        st.write(f"• Records: {len(filtered_df):,}")
        st.write(f"• Columns: {len(filtered_df.columns)}")
        st.write(f"• Size: ~{len(filtered_df) * len(filtered_df.columns) * 50 / 1024:.1f} KB")

    if st.button("📥 Generate Download", type="primary"):
        try:
            if export_format == "CSV":
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download CSV", 
                    csv_data, 
                    f"{filename}.csv", 
                    "text/csv",
                    key="download_csv"
                )
            elif export_format == "Excel":
                # Note: This requires openpyxl or xlsxwriter
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name='Reviews', index=False)
                st.download_button(
                    "📥 Download Excel", 
                    buffer.getvalue(), 
                    f"{filename}.xlsx", 
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            elif export_format == "JSON":
                json_data = filtered_df.to_json(orient='records', indent=2).encode('utf-8')
                st.download_button(
                    "📥 Download JSON", 
                    json_data, 
                    f"{filename}.json", 
                    "application/json",
                    key="download_json"
                )
            st.success("✅ Download ready!")
        except Exception as e:
            st.error(f"⚠️ Export error: {str(e)}")
else:
    st.warning("⚠️ No data to export. Please adjust filters.")

# View raw filtered data
with st.expander("📄 View Filtered Data"):
    if not filtered_df.empty:
        st.write(f"**Showing {len(filtered_df):,} rows**")
        
        # Add column selector
        if len(filtered_df.columns) > 5:
            selected_columns = st.multiselect(
                "Select columns to display:", 
                filtered_df.columns.tolist(), 
                default=filtered_df.columns.tolist()[:5]
            )
            if selected_columns:
                display_df = filtered_df[selected_columns]
            else:
                display_df = filtered_df
        else:
            display_df = filtered_df
        
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("No data matches the current filters.")

# Debug information (expandable)
with st.expander("🔧 Debug Information"):
    st.markdown("**Dataset Overview:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Data:**")
        st.write(f"• Shape: {df.shape}")
        st.write(f"• Columns: {list(df.columns)}")
        if "Category" in df.columns:
            st.write(f"• Unique Categories: {df['Category'].nunique()}")
    
    with col2:
        st.write("**Filtered Data:**")
        st.write(f"• Shape: {filtered_df.shape}")
        st.write(f"• Apps Selected: {len(selected_apps)}")
        st.write(f"• Categories Selected: {len(selected_categories)}")
        st.write(f"• Sentiments Selected: {len(selected_sentiments)}")
    
    if "Category" in df.columns:
        st.markdown("**All Categories in Dataset:**")
        all_cats = sorted(df["Category"].unique())
        st.write(f"Total: {len(all_cats)}")
        st.write(all_cats)

# Footer
st.markdown("---")
st.markdown("*Dashboard created with Streamlit • Data insights for better product decisions*")