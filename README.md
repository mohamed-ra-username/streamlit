# 🛰️ Meteor Bot – NLP Analysis Suite

Streamlit application that instantly turns any chunk of text into data-rich insights:

* **Topic detection (zero-shot, multi-label)**
* **Sentiment analysis with emoji cue**
* **Automatic summary (DistilBART)**
* **Text statistics & word-cloud**
* **Confidence bar-chart for topics**
* **Chat-style UI, dark-mode theme, “Basic / Advanced” view toggle**
* **Natural small-talk replies (hi / help / thanks / bye)**
* **Demo:**
    ```bash
    streamlit run streamlined_meteor_bot.py
    ```
    First launch downloads ≈ 300 MB of models; subsequent runs start instantly

## Feature Matrix

| Capability           | Model / Library                 | Notes                                    |
| :------------------- | :------------------------------ | :--------------------------------------- |
| Topic detection      | valhalla/distilbart-mnli-12-3    | Zero-shot, multi-label                   |
| Summarisation        | sshleifer/distilbart-cnn-12-6   | Abstractive                              |
| Sentiment analysis   | distilbert-base-uncased-finetuned-sst-2 | POS / NEG score                        |
| Word-cloud           | wordcloud Python lib            | Common-word filter                       |
| Visualisations       | matplotlib, Streamlit native charts |                                          |
| Dark-mode colours    |                                 |                                          |
| Caching              | `@st.cache_resource`            | Models loaded once per run               |

## Installation

```bash
git clone <repo-url>
cd meteor-bot
python -m venv venv          # optional but recommended
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlined_meteor_bot.py
