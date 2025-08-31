🇦🇺 CommunityMate — Connecting New Citizens to Australian Democracy (Hackathon Demo)
CommunityMate is a Streamlit app prototype that helps new Australians:

Discover and access local government and community services
Learn about democracy and civics (using Parliamentary Education Office (PEO) content)
Get travel times, reminders, and demo bookings for services
👉 This is a hackathon demo, so bookings, reminders, and form submissions are simulated only and stored locally in the out/ folder.

🚀 Quick Start
1. Clone this repo
git clone <your-repo-url>
cd HACK2

2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate # On macOS/Linux
venv\Scripts\activate # On Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the Streamlit app
streamlit run app.py

Then open your browser to:
http://localhost:8501

📂 Project Structure
HACK2/
│
├── app.py # Main Streamlit app (UI + chat agent)
├── communitymate.py # Core logic (indexes, tools, service handling)
├── datacrawler.py # PEO civics crawler (see below)
├── data/ # Holds crawled civics dataset + services CSV
├── out/ # Generated reminders, receipts, form submissions
├── requirements.txt # Dependencies
├── logo.jpeg # Logo displayed in the sidebar
└── README.md # This file

📚 Data
Service Data: CSV files in data/ with government & community services.
Civics Knowledge Base: Pre‑crawled PEO content stored at data/peo/manifest.jsonl.
The repo already includes data, so you can run the app immediately.

If you want to refresh or re‑crawl civics resources:
python datacrawler.py

This script:

Crawls Parliamentary Education Office (PEO) civics content
Saves results into data/peo/manifest.jsonl
Used by the app to provide retrieval‑augmented answers with citations.
