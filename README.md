# Real-Time Sentiment Analysis Dashboard

A streaming NLP pipeline that ingests live social media data, performs multi-class sentiment analysis using a fine-tuned transformer, and visualizes results in a real-time Plotly Dash dashboard.

## Features
- Real-time Twitter/Reddit stream ingestion via API v2
- Transformer-based sentiment classification (positive / negative / neutral / mixed)
- Aspect-based sentiment analysis (ABSA) per entity/topic
- Named Entity Recognition (NER) for topic extraction
- Live Plotly Dash dashboard with streaming updates
- Kafka message queue for scalable ingestion
- Redis caching for deduplication

## Tech Stack
`Python` `Transformers` `Dash/Plotly` `Kafka` `Redis` `Tweepy` `spaCy` `Docker`

## Architecture

```
Twitter/Reddit API
      ↓
  Kafka Topic
      ↓
Sentiment Processor (Transformer)
      ↓
  Redis Cache ──→ Dash Dashboard (WebSocket)
      ↓
  PostgreSQL (historical)
```

## Quickstart

```bash
# Start infrastructure
docker-compose up -d kafka redis postgres

# Start the ingestion pipeline
python pipeline/ingest.py --source twitter --keywords "AI,MachineLearning,DataScience"

# Start sentiment processor
python pipeline/processor.py --workers 4

# Launch dashboard
python dashboard/app.py
# Visit http://localhost:8050
```

## Project Structure

```
├── pipeline/
│   ├── ingest.py           # Kafka producer: Twitter/Reddit stream
│   ├── processor.py        # Kafka consumer: sentiment classification
│   └── storage.py          # PostgreSQL + Redis writer
├── models/
│   ├── sentiment.py        # Fine-tuned RoBERTa sentiment classifier
│   ├── ner.py              # spaCy NER for topic extraction
│   └── absa.py             # Aspect-based sentiment analysis
├── dashboard/
│   ├── app.py              # Dash app entrypoint
│   ├── layouts.py          # Dashboard layout components
│   └── callbacks.py        # Real-time update callbacks
├── docker-compose.yml
└── requirements.txt
```

## Sentiment Classifier (`models/sentiment.py`)

```python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentAnalyzer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            batch_size=32,
        )
        self.label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        results = self.pipe(texts, truncation=True, max_length=512)
        return [
            {
                "label": self.label_map.get(r["label"], r["label"]),
                "score": round(r["score"], 4),
            }
            for r in results
        ]

    def analyze_with_aspects(self, text: str, aspects: list[str]) -> dict:
        """Aspect-based sentiment: score sentiment per named aspect."""
        overall = self.analyze_batch([text])[0]
        aspect_sentiments = {}
        for aspect in aspects:
            context = f"{aspect}: {text}"
            aspect_sentiments[aspect] = self.analyze_batch([context])[0]
        return {"overall": overall, "aspects": aspect_sentiments}
```

## Dashboard Preview

The Dash dashboard includes:
- **Live feed**: scrolling list of incoming posts with sentiment badges
- **Sentiment gauge**: real-time positive/negative/neutral ratio
- **Time series**: sentiment trend over sliding 1-hour window
- **Word cloud**: top terms per sentiment class
- **Topic heatmap**: sentiment by topic × time matrix

## Configuration

```yaml
# config.yml
kafka:
  bootstrap_servers: "localhost:9092"
  topic: "social-stream"
  group_id: "sentiment-processor"

redis:
  host: localhost
  port: 6379
  ttl: 3600

model:
  name: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  batch_size: 32
  confidence_threshold: 0.7

dashboard:
  update_interval_ms: 2000
  window_size: 500
```

## References
- [RoBERTa for Sentiment Analysis](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Kafka Python Client](https://kafka-python.readthedocs.io/)
- [Plotly Dash Documentation](https://dash.plotly.com/)
