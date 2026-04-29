# 🏠 Real Estate Advisor – RERA Legal Guide
### Mini Project | Agentic AI Application Development using LangChain Framework
**Department of CSE | HKBK College of Engineering | 6A – Batch 15**

---

## 👥 Team Members
| USN | Name |
|---|---|
| 1HK23CS059 | Kalluri Indraja |
| 1HK23CS060 | Kavali Manoj Kumar |
| 1HK23CS061 | Kaviya S |
| 1HK23CS062 | Keerthana KH |

---

## 📌 Problem Statement
Property buyers in India are often unaware of their legal rights under RERA, leading to financial losses due to builder fraud, delayed possession, and non-compliance with regulations. This project builds an AI-powered Real Estate Advisor that guides buyers through:
- Local property laws under **RERA (Real Estate Regulation and Development Act, 2016)**
- **Legal must-check checklist** before purchasing a property
- **Buyer's registration documents** and stamp duty information
- **Grievance redressal** rights and procedures

---

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| Framework | LangChain (RAG Pipeline) |
| LLM | Groq – LLaMA 3.1 8B Instant (free tier) |
| Embeddings | HuggingFace – `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| UI | Streamlit |
| Knowledge Base | Custom RERA text document |

---

## 🏗️ Architecture (RAG Pipeline)

```
User Query
    ↓
[HuggingFace Embeddings]  →  Query Vector
    ↓
[ChromaDB Vector Store]   →  Top-4 Relevant Chunks  (Retrieval)
    ↓
[ChatPromptTemplate]      →  System Prompt + Context + User Question
    ↓
[Groq LLaMA 3.1 8B]      →  Grounded Answer         (Generation)
    ↓
Streamlit UI              →  Displayed to User
```

---

## 📁 Project Structure
```
real_estate_project/
├── real_estate_advisor.py   ← Main Streamlit application
├── rera_knowledge.txt       ← RERA knowledge base (RAG document)
├── requirements.txt         ← Python dependencies
├── .env                     ← API keys (not committed)
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone / Download the project
```bash
cd real_estate_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at: https://console.groq.com

### 4. Run the app
```bash
streamlit run real_estate_advisor.py
```

---

## 💡 Features
- ✅ Ask any RERA / property law question in natural language
- ✅ Quick-access buttons for the most common buyer queries
- ✅ Shows source context chunks used for each answer (transparent RAG)
- ✅ Persistent chat history within a session
- ✅ Sidebar with team info and tech stack
- ✅ Built entirely on **free APIs** (Groq free tier + HuggingFace)

---

## 📖 Sample Questions to Try
1. What is RERA and who needs to register?
2. What documents should I verify from the builder before buying?
3. What are my rights if the builder delays possession?
4. What is carpet area under RERA?
5. What is the stamp duty and registration charge in Karnataka?
6. How do I file a complaint against a builder on RERA portal?
7. What is an Occupation Certificate and why is it important?
8. What is an encumbrance certificate?

---

## 🔗 References
- RERA Karnataka: https://rera.karnataka.gov.in
- MahaRERA: https://maharera.maharashtra.gov.in
- Real Estate (Regulation and Development) Act, 2016
- LangChain Documentation: https://python.langchain.com
