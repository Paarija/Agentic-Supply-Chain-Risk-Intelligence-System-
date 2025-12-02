# Agentic Supply Chain Risk Intelligence System  
### Multi-Agent AI • Streamlit App • CrewAI • Gemini 2.5 • Live Risk Scanning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-green)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-yellow)


The system analyzes supplier delays, identifies high-risk vendors, scans external news for disruptions, and generates a **strategic risk report**—all in one automated workflow.

---

# Features

###  **1. Internal Delay & Risk Analysis**
- Upload a CSV containing supplier delay data  
- Automatic detection of suppliers with:
  - **avg_delay_days > 5**, or  
  - **risk_score > 50**
- Highlights all risky suppliers

###  **2. External News Investigation (AI Agent)**
- AI agent queries Google News for each risky supplier's location  
- Detects:
  - local strikes  
  - weather disruptions  
  - political instability  
  - transport issues  
- Falls back to baseline reasoning when news is unavailable  
  (real-world limitation turned into a feature)

###  **3. Multi-Agent AI Workflow**
- **Auditor Agent** → analyzes CSV  
- **Risk Investigator Agent** → researches external risks  
- **Manager Agent** → produces final actionable recommendations  
- Powered by **CrewAI**

###  **4. Auto-Generated Strategic Report**
Output includes:
- Supplier risk summary  
- External news insights  
- Action plan (retain / monitor / replace)  
- Markdown format, exportable to PDF

###  **5. Clean & Interactive Streamlit Interface**
- Upload CSV  
- Preview dataset  
- Run full agent pipeline  
- Get the results instantly

---
# Demo
<img width="1366" height="626" alt="Screenshot (1127)" src="https://github.com/user-attachments/assets/83530342-d64b-414d-9f89-5e2b8cac0496" />
<img width="1366" height="624" alt="Screenshot (1128)" src="https://github.com/user-attachments/assets/74ba1457-b08f-4164-84e8-23f1ecb70d73" />
<img width="1366" height="626" alt="Screenshot (1129)" src="https://github.com/user-attachments/assets/6fa575a0-41bf-4524-9866-5243c513a0c4" />



# 📁 Sample Dataset Structure

Your internal_suppliers.csv should contain:

| supplier_id | supplier_name | location | avg_delay_days | risk_score |
|-------------|----------------|-----------|------------------|-------------|
| 1 | Flashpoint | Gunajaya | 11 | 42 |
| 2 | Yodoo | Kitui | 10 | 71 |
| ... | ... | ... | ... | ... |

*(This project uses a synthetic dataset generated via Mockaroo to simulate real supplier behavior.)*

---

#  How to Run Locally

```bash
# 1. Clone repo
git clone https://github.com/Paarija/Agentic-Supply-Chain-Risk-Intelligence-System.git
cd Agentic-Supply-Chain-Risk-Intelligence-System
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
