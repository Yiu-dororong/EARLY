# early-system
EARLY


Planned Structure:

```
early/
├──agents/
│   ├── __init__.py
│   ├── orchestrator.py       
│   ├── sentiment_auditor.py  
│   ├── forensic_agent.py     
│   └── critic_agent.py      
│ 
├── api/                          # FastAPI serving layer
│   ├── main.py
│   ├── routers/
│   │   ├── games.py            # GET /games/{appid}, POST /games/{appid}/analyse
│   │   └── health.py
│   ├── services/
│   │   ├── inference.py
│   │   ├── scorecard.py
│   │   └── agents.py           # Thin adapter: calls agents.orchestrator.run_analysis()
│   └── utils/
│       ├── db.py
│       └── cache.py
│
├── core/                         # Core business logic (shared)
│   ├── builders/
│   │   ├── build_snapshots.py
│   │   └── feature_builder.py 
│   ├── inference/
│   │   └── inference.py            
│   └── schemas/                   # XGBoost artifacts
│
├── data/                         # Data pipeline scripts
│   ├── collection/               # All collect_*.py files
│   │   ├── collect_ccu_history.py  
│   │   ├── collect_events.py       
│   │   ├── collect_genres.py       
│   │   ├── collect_review_history.py 
│   │   ├── collect_pre2022_ea_games.py 
│   │   └── pipeline_discovery.py   # Initial routing and target acquisition for the scraping 
│   └── processing/               
│       ├── compute_dev_features.py     
│       ├── compute_genre_price_median    
│       └── label_outcomes.py       
│
├── evaluation/                   # Training & evaluation
│   ├── train_xgboost.py
│   ├── scorecard.py
│   ├── scorecard_config.py
│   ├── scorecard_evaluate.py
│   └── outputs/
│
├── frontend/                     # Streamlit UI
│   └── app.py
│
├── infrastructure/               # Deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env.example
│
├── utils/                 # Small utilities
│   └── itad_client.py
│
├── models/                       # ML artifacts
├── README.md
├── requirements.txt
├── .env
└── .gitignore
```