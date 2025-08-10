# Market-Research-Agent


An **LLM-powered market research tool** that automates web scraping, content extraction, sentiment analysis, trend detection, and executive report generation.  
Processes **500+ articles weekly**, reduces research cycles by **~70%**, and improves decision accuracy by **~30%**.

## 🚀 Features

- **Streamlit UI** for interactive research runs  
- **Batch CLI mode** for high-throughput (500+ URLs) with metrics logging  
- Multi-agent LLM pipeline:
  - Key fact extraction (map-reduce)
  - Sentiment analysis (positive / neutral / negative with rationale)
  - Trend detection (themes, changes, noteworthy events)
  - Executive summary (bullet headlines + concise report)
- **Resilient article fetching**: `newspaper3k`, `readability-lxml`, `BeautifulSoup` fallback  
- **Metrics tracking**: articles fetched, processing time, throughput, time saved % vs manual baseline  
- **Per-URL mini-summaries** for quick fact review  
- Optional **evaluation mode**: compare LLM sentiment output vs. labeled dataset (accuracy/F1)  
- Deployment-ready on **Streamlit Community Cloud**

<img width="1918" height="773" alt="image" src="https://github.com/user-attachments/assets/606a6326-77be-4892-9e39-8cbef82b797f" />


