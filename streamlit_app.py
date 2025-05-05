# streamlined_meteor_bot.py
# =============================================================
#  🔭 Meteor Bot – NLP Analysis Suite (Streamlit)
# =============================================================
import os
import random
import streamlit as st
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# --- make sure Hugging Face never calls TensorFlow / Keras ----------
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import pipeline

# ---------- Lazy‑load + cache NLP pipelines ---------------------
@st.cache_resource(show_spinner="⏳ Downloading models... (first run only) ⌛")
def load_pipelines():
    """Load and cache NLP pipelines for analysis"""
    try:
        # Distilled MNLI for zero‑shot topics (≈ 230 MB)
        classifier = pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=-1,  # CPU
        )
        
        # Distilled CNN summarizer (≈ 80 MB)
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1,
        )
        
        # Sentiment analysis pipeline
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
        )
        
        # Warm‑up pass so first user is fast
        classifier("warm‑up", ["test"])
        sentiment_analyzer("This is a test")
        # Use appropriate parameters for warm-up to avoid warnings
        summarizer("This is a longer warm-up text to avoid length warnings.", 
                 max_length=10, min_length=5, do_sample=False)
                 
        return classifier, summarizer, sentiment_analyzer
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        # Return fallback functions
        def fallback_classifier(text, labels, **kwargs):
            return {"labels": labels, "scores": [0.5] * len(labels)}
        
        def fallback_summarizer(text, **kwargs):
            return [{"summary_text": "Could not generate summary due to model loading error."}]
            
        def fallback_sentiment(text, **kwargs):
            return [{"label": "NEUTRAL", "score": 0.5}]
            
        return fallback_classifier, fallback_summarizer, fallback_sentiment

# --------------- Topic list -----------------------------------------
TOPICS = [
    "politics", "business", "finance", "technology", "science", "health", "sports",
    "entertainment", "music", "movies", "education", "environment", "travel",
    "food", "fashion", "art", "history", "literature", "space", "gaming",
    "psychology", "philosophy", "social media", "artificial intelligence", "climate",
]

# --------------- Small‑talk shortcuts -------------------------------
HELLOS = {"hi", "hello", "hey", "yo", "sup", "greetings", "what's up"}
THANKS = {"thanks", "thank you", "thx", "ty"}
BYES = {"bye", "goodbye", "see you", "cya", "farewell"}
HELP = {"help", "?", "how", "what can you do", "instructions"}

def quick_reply(text: str):
    """Handle common conversation starters/enders"""
    t = text.lower().strip()
    if t in HELLOS:
        return "👋 Welcome to Meteor Bot! Paste any paragraph and I'll analyze its topics, sentiment & generate a summary."
    if t in THANKS:
        return "You're welcome! 😊 Happy to help anytime."
    if t in BYES:
        return "Goodbye! Come back soon!"
    if t in HELP:
        return (
            "📚 **How to use Meteor Bot:**\n\n"
            "1. Paste a paragraph or article (at least 6 words)\n"
            "2. I'll automatically identify key topics and perform sentiment analysis\n"
            "3. I'll generate a concise summary\n"
            "4. You can view additional text statistics and visualizations\n\n"
            "You can adjust the number of topics and summary length in the sidebar."
        )
    return None

# --------------- Basic NLP Analysis Functions --------------------------------------
def get_text_stats(text):
    """Calculate basic text statistics"""
    words = text.split()
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    stats = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": text.count('\n\n') + 1,
        "avg_word_length": sum(len(word) for word in words) / max(1, len(words)),
        "avg_sentence_length": len(words) / max(1, len(sentences)),
    }
    
    return stats

def generate_wordcloud(text):
    """Generate wordcloud from text after basic cleaning"""
    # Basic tokenization and cleaning
    common_words = set(['the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'that', 'for', 
                        'with', 'as', 'be', 'this', 'on', 'not', 'by', 'at', 'from', 'or', 
                        'an', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'they', 'we'])
    
    # Clean and tokenize
    words = text.lower().split()
    words = [word.strip('.,!?:;()[]{}""''') for word in words]
    words = [word for word in words if word and word not in common_words and len(word) > 2]
    
    word_freq = Counter(words)
    
    # Create and configure the WordCloud object
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='#192231',
        colormap='viridis',
        max_words=100,
        contour_width=3,
        contour_color='#3F4E5E'
    ).generate_from_frequencies(word_freq)
    
    return wordcloud

# --------------- Analysis core --------------------------------------
def analyze(text: str, topic_threshold=0.2, max_topics=3, summary_min_length=15, summary_max_length=45):
    """Analyze text using multiple NLP techniques"""
    # Load models if not already loaded
    cls_pipe, sum_pipe, sentiment_pipe = load_pipelines()
    results = {}
    
    with st.spinner("🔍 Analyzing topics..."):
        # Get topic classification
        z = cls_pipe(text, TOPICS, multi_label=True)
        
        # Take labels above threshold, keep max number
        chosen = [l for l, s in zip(z["labels"], z["scores"]) if s >= topic_threshold][:max_topics] \
                 or [z["labels"][0]]
        
        # Get scores for chosen topics
        scores = {l: round(s * 100) for l, s in zip(z["labels"], z["scores"]) if l in chosen}
        results["topics"] = {"labels": chosen, "scores": scores}
    
    with st.spinner("😀 Analyzing sentiment..."):
        # Get sentiment analysis
        sentiment = sentiment_pipe(text)[0]
        results["sentiment"] = {
            "label": sentiment["label"],  
            "score": round(sentiment["score"] * 100)
        }
        
    with st.spinner("📊 Calculating statistics..."):
        # Get text statistics
        stats = get_text_stats(text)
        results["stats"] = stats
        
        # Generate wordcloud
        wordcloud = generate_wordcloud(text)
        results["wordcloud"] = wordcloud
        
    with st.spinner("📝 Generating summary..."):
        # Calculate appropriate max length based on input text
        input_length = len(text.split())
        adjusted_max_length = min(summary_max_length, max(summary_min_length, input_length - 2))
        
        # Generate summary
        summary = sum_pipe(
            text, 
            max_length=adjusted_max_length, 
            min_length=min(summary_min_length, adjusted_max_length - 5), 
            do_sample=False
        )[0]["summary_text"]
        results["summary"] = summary
        
    return results

def bot_reply(user_text: str, topic_threshold=0.2, max_topics=3, summary_min_length=15, summary_max_length=45):
    """Generate a bot reply based on user input"""
    # Check for quick replies first
    qr = quick_reply(user_text)
    if qr:
        return qr, None
    
    # Check for minimum text length
    if len(user_text.split()) < 6:
        return "⚠️ Please enter at least a few sentences for accurate analysis.", None
    
    # Analyze text
    results = analyze(
        user_text, 
        topic_threshold=topic_threshold, 
        max_topics=max_topics,
        summary_min_length=summary_min_length,
        summary_max_length=summary_max_length
    )
    
    # Format topics with confidence scores
    topic_text = ", ".join([f"{t} ({results['topics']['scores'][t]}%)" for t in results['topics']['labels']])
    
    # Format sentiment
    sentiment_emoji = "😃" if results['sentiment']['label'] == "POSITIVE" else "😐" if results['sentiment']['label'] == "NEUTRAL" else "😟"
    sentiment_text = f"{sentiment_emoji} {results['sentiment']['label']} ({results['sentiment']['score']}%)"
    
    reply = f"**Topics:** {topic_text}\n\n**Sentiment:** {sentiment_text}\n\n**Summary:** {results['summary']}"
    
    return reply, results

# --------------- Streamlit UI ---------------------------------------
# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Meteor Bot | NLP Analysis", 
    page_icon="🔭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #192231;
        color: #e6e6e6;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #15202b;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #c6d5eb !important;
    }
    
    /* Text input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #253446;
        color: #e6e6e6;
        border-color: #3F4E5E;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #3F4E5E;
    }
    .stSlider > div > div > div > div {
        background-color: #6497b1;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #253446;
        color: #c6d5eb;
        border-color: #3F4E5E;
    }
    .streamlit-expanderContent {
        background-color: #192231;
        border-color: #3F4E5E;
    }
    
    /* Divider */
    hr {
        border-color: #3F4E5E;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #253446;
        border-left: 4px solid #6497b1;
    }
    .chat-message.bot {
        background-color: #1a2734;
        border-left: 4px solid #4e7ca8;
    }
    .chat-icon {
        width: 50px;
        font-size: 24px;
        text-align: center;
    }
    .chat-content {
        flex-grow: 1;
        padding-left: 1rem;
        color: #e6e6e6;
    }
    .timestamp {
        font-size: 12px;
        color: #a0a0a0;
        margin-top: 4px;
    }
    
    /* Button styles */
    .stButton > button {
        background-color: #4e7ca8;
        color: white;
        border: none;
        border-radius: 4px;
    }
    .stButton > button:hover {
        background-color: #6497b1;
    }
    
    /* Tables */
    .stTable {
        background-color: #253446;
        color: #e6e6e6;
    }
    .stDataFrame {
        background-color: #253446;
    }
    
    /* Charts */
    .stChart > div > div > svg {
        background-color: #192231 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #253446;
        color: #c6d5eb;
    }
    .highlight {
        background-color: #253446;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #253446;
    }
    .stMetric > div {
        background-color: #253446;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4e7ca8;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #15202b;
    }
    .stTabs [data-baseweb="tab"] {
        color: #c6d5eb;
    }
    .stTabs [aria-selected="true"] {
        background-color: #253446;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("🔭 Meteor Bot")
        st.markdown("#### Text Analysis Settings")
        
        analysis_tab = st.radio(
            "Analysis Mode",
            ["Basic", "Advanced"],
            help="Choose analysis depth"
        )
        
        topic_threshold = st.slider(
            "Topic confidence threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.2, 
            step=0.05,
            help="Minimum confidence score for a topic to be included"
        )
        
        max_topics = st.slider(
            "Maximum topics to display", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Maximum number of topics to show in results"
        )
        
        st.markdown("#### Summary Settings")
        summary_min_length = st.slider(
            "Minimum summary length", 
            min_value=10, 
            max_value=30, 
            value=15,
            help="Minimum word count for generated summary"
        )
        
        summary_max_length = st.slider(
            "Maximum summary length", 
            min_value=30, 
            max_value=100, 
            value=45,
            help="Maximum word count for generated summary"
        )
        
        if st.button("Clear Chat History", type="primary"):
            st.session_state.chat = []
            st.session_state.analysis_results = []
            st.rerun()
            
        st.markdown("---")
        st.markdown("""
        **Group members**
        - Ahmed Mohammed Diab  
        - Mohammed Elsaeed Abdelfatah Shalaby  
        - Mohammed Ragab Mubarak  
        - Youssef Mohammed Youssef
        """)
    
    # Main content area
    st.title("🔭 Meteor Bot – NLP Analysis Suite")
    st.markdown(
        "Paste any text to automatically detect topics, sentiment and generate a concise summary."
    )
    
    # Initialize session state for chat history and analysis results
    if "chat" not in st.session_state:
        st.session_state.chat = []
        st.session_state.analysis_results = []
        # Add welcome message
        st.session_state.chat.append((
            "Bot", 
            "👋 Welcome to Meteor Bot! Paste any paragraph and I'll analyze its topics, sentiment & generate a summary.",
            datetime.now().strftime("%H:%M")
        ))
        st.session_state.analysis_results.append(None)
    
    # Function to add messages to chat history
    def add_msg(role: str, text: str, results=None):
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.chat.append((role, text, timestamp))
        st.session_state.analysis_results.append(results)
    
    # Function to handle form submission
    def handle_submit():
        txt = st.session_state.user_input.strip()
        st.session_state.user_input = ""  # clear box
        if not txt:
            return
        
        add_msg("User", txt)
        
        # Generate and add bot response
        response, results = bot_reply(
            txt,
            topic_threshold=topic_threshold,
            max_topics=max_topics,
            summary_min_length=summary_min_length,
            summary_max_length=summary_max_length
        )
        add_msg("Bot", response, results)
    
    # Chat display
    chat_col, viz_col = st.columns([3, 2]) if analysis_tab == "Advanced" else [st.container(), None]
    
    with chat_col:
        for i, ((role, text, timestamp), results) in enumerate(zip(st.session_state.chat, st.session_state.analysis_results)):
            if role == "User":
                st.markdown(f"""
                    <div class="chat-message user">
                        <div class="chat-icon">👤</div>
                        <div class="chat-content">
                            <b>You</b>
                            <div>{text}</div>
                            <div class="timestamp">{timestamp}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot">
                        <div class="chat-icon">🔭</div>
                        <div class="chat-content">
                            <b>Meteor Bot</b>
                            <div>{text}</div>
                            <div class="timestamp">{timestamp}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display additional analysis if available and we're in advanced mode
                if analysis_tab == "Advanced" and results and viz_col and i == len(st.session_state.chat) - 1:
                    with viz_col:
                        st.subheader("NLP Analysis Details")
                        
                        # Text statistics
                        if "stats" in results:
                            with st.expander("📊 Text Statistics", expanded=True):
                                stats = results["stats"]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Word Count", f"{stats['word_count']}")
                                with col2:
                                    st.metric("Sentences", f"{stats['sentence_count']}")
                                with col3:
                                    st.metric("Paragraphs", f"{stats['paragraph_count']}")
                                
                                if "avg_sentence_length" in stats:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
                                    with col2:
                                        st.metric("Avg Sentence Length", f"{stats['avg_sentence_length']:.1f}")
                        
                        # Word cloud
                        if "wordcloud" in results and results["wordcloud"]:
                            with st.expander("☁️ Word Cloud", expanded=True):
                                # Create a figure and axis
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(results["wordcloud"], interpolation='bilinear')
                                ax.axis("off")
                                ax.set_facecolor('#192231')
                                fig.patch.set_facecolor('#192231')
                                
                                # Display the figure
                                st.pyplot(fig)
                        
                        # Topics visualization
                        if "topics" in results and results["topics"]["labels"]:
                            with st.expander("📌 Topic Confidence", expanded=True):
                                # Create dataframe for bar chart
                                topic_df = pd.DataFrame({
                                    'Topic': results["topics"]["labels"],
                                    'Confidence': [results["topics"]["scores"][label] for label in results["topics"]["labels"]]
                                })
                                
                                # Sort by confidence
                                topic_df = topic_df.sort_values('Confidence', ascending=False)
                                
                                # Create bar chart
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.barh(topic_df['Topic'], topic_df['Confidence'], color='#4e7ca8')
                                ax.set_xlabel('Confidence (%)')
                                ax.set_ylabel('Topic')
                                ax.set_facecolor('#192231')
                                ax.tick_params(colors='#e6e6e6')
                                ax.xaxis.label.set_color('#e6e6e6')
                                ax.yaxis.label.set_color('#e6e6e6')
                                fig.patch.set_facecolor('#192231')
                                for spine in ax.spines.values():
                                    spine.set_color('#3F4E5E')
                                
                                st.pyplot(fig)
    
    # Input area with form for better control
    with st.form(key="input_form", clear_on_submit=False):
        st.text_area(
            "Type your text here",
            key="user_input",
            height=150,
            placeholder="Paste a paragraph or article here (at least 6 words)..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submit_button = st.form_submit_button(
                label="Analyze", 
                type="primary",
                use_container_width=True,
                on_click=handle_submit
            )
        with col2:
            clear_button = st.form_submit_button(
                label="Clear Input", 
                type="secondary",
                use_container_width=True,
                on_click=lambda: setattr(st.session_state, "user_input", "")
            )

if __name__ == "__main__":
    # Run the app
    main()
