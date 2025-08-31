# 🇦🇺 CommunityMate — Australian Government & Community Services Assistant (Hackathon Demo)
## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the app
```bash
streamlit run app.py
```

The app will open in your browser at:
```
http://localhost:8501
```

---

## 🏗️ Architecture:

The diagram below shows how **CommunityMate** works end‑to‑end:

<img width="3351" height="1584" alt="Copy of Untitled Diagram drawio (1)" src="https://github.com/user-attachments/assets/04686b93-ad54-46dc-8fbb-a18d9530f6b7" />


### 🔑 Components
- **User (Web Browser)** → interacts with **Streamlit Chat UI**  
- **Agent (LangChain + OpenAI)** → interprets queries and decides which tool to use  
- **Tools/Modules**:  
  - 🗂️ **Service Index (CSV + FAISS)** → finds relevant government & community services  
  - 🧠 **Logic & Consent Engine** → ensures safety, scope limits, and explicit consent  
  - 🗺️ **Travel Estimator (OSRM + heuristics)** → driving, walking, transit times  
  - 📝 **Booking & Reminder Simulations** → local receipts/files saved in `out/`  
