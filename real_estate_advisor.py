import os
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
import numpy as np

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ─────────────────────────────
# UI CONFIG
# ─────────────────────────────
st.set_page_config(page_title="Real Estate AI", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { width: 380px; }
body { background-color: #0e1117; }
h1, h2 { color: #00ffd5; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 Real Estate AI System")

# ─────────────────────────────
# RAG FUNCTION
# ─────────────────────────────
@st.cache_resource
def build_rag(docs_list):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs_list)

    db = Chroma.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Real Estate Legal Advisor.
Use ONLY the given context.

Context:
{context}
"""),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, chain)

# Default chatbot
default_docs = TextLoader("rera_knowledge.txt").load()
main_rag = build_rag(default_docs)

# ─────────────────────────────
# SIDEBAR
# ─────────────────────────────
st.sidebar.title("⚡ Features")

feature = st.sidebar.radio(
    "Select",
    [
        "💬 AI Chat",
        "📄 Upload PDF",
        "📍 Smart Map",
        "📊 Dashboard",
        "⚠️ Risk Analyzer",
        "💰 Cost Estimator",
        "🤖 Price Prediction"
    ]
)

# Suggested questions
st.sidebar.markdown("### 💡 Suggested Questions")
suggested = [
    "What is RERA?",
    "Documents needed before buying property?",
    "How to verify builder?",
    "Stamp duty details?"
]

selected_q = None
for q in suggested:
    if st.sidebar.button(q):
        selected_q = q

# ─────────────────────────────
# CHAT
# ─────────────────────────────
if feature == "💬 AI Chat":
    st.header("💬 Ask AI")

    query = st.text_input("Ask anything...")
    query = selected_q or query

    if query:
        res = main_rag.invoke({"input": query})
        st.success(res["answer"])

# ─────────────────────────────
# PDF (FIXED)
# ─────────────────────────────
elif feature == "📄 Upload PDF":
    st.header("📄 Upload & Ask PDF")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        file_path = "temp_uploaded.pdf"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            pdf_rag = build_rag(docs)

            st.success("✅ PDF loaded successfully!")

            query = st.text_input("Ask question from PDF")

            if query:
                with st.spinner("Analyzing PDF..."):
                    res = pdf_rag.invoke({"input": query})
                    st.write(res["answer"])

        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────
# 🔥 FIXED SMART MAP
# ─────────────────────────────
elif feature == "📍 Smart Map":
    st.header("📍 Smart Property Map")

    data = pd.DataFrame({
        "area": ["Whitefield", "BTM", "Indiranagar", "Electronic City"],
        "lat": [12.9698, 12.9166, 12.9784, 12.8399],
        "lon": [77.7500, 77.6101, 77.6408, 77.6770],
        "price": [80, 60, 120, 55]
    })

    def get_color(price):
        if price > 100:
            return [255, 0, 0]
        elif price > 70:
            return [255, 165, 0]
        else:
            return [0, 255, 0]

    data["color"] = data["price"].apply(get_color)

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=12.95,
            longitude=77.65,
            zoom=11,
            pitch=30
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=data,
                get_position='[lon, lat]',
                get_color="color",
                get_radius=500,
                pickable=True
            )
        ],
        tooltip={
            "html": "<b>{area}</b><br/>Price: ₹{price} Lakhs",
            "style": {"color": "black"}
        }
    ))

    st.markdown("🔴 Expensive | 🟠 Medium | 🟢 Affordable")

# ─────────────────────────────
# DASHBOARD
# ─────────────────────────────
elif feature == "📊 Dashboard":
    st.header("📊 Market Trends")

    df = pd.DataFrame({
        "Area": ["Whitefield", "BTM", "Indiranagar"],
        "Price": [80, 60, 120]
    })

    fig = px.bar(df, x="Area", y="Price")
    st.plotly_chart(fig)

# ─────────────────────────────
# RISK ANALYZER
# ─────────────────────────────
elif feature == "⚠️ Risk Analyzer":
    st.header("⚠️ Risk Analyzer")

    rera = st.selectbox("RERA?", ["Yes", "No"])
    bank = st.selectbox("Bank Approved?", ["Yes", "No"])
    title = st.selectbox("Clear Title?", ["Yes", "No"])

    if st.button("Analyze"):
        if "No" in [rera, bank, title]:
            st.error("🚨 High Risk")
        else:
            st.success("✅ Safe")

# ─────────────────────────────
# COST
# ─────────────────────────────
elif feature == "💰 Cost Estimator":
    st.header("💰 Cost Estimator")

    price = st.number_input("Enter price ₹")

    if st.button("Calculate"):
        st.write(f"Stamp Duty: ₹{price*0.05:,.2f}")
        st.write(f"Registration: ₹{price*0.01:,.2f}")

# ─────────────────────────────
# ML
# ─────────────────────────────
elif feature == "🤖 Price Prediction":
    st.header("🤖 AI Price Prediction")

    X = np.array([[1000], [1500], [2000], [2500]])
    y = np.array([50, 70, 90, 120])

    model = LinearRegression()
    model.fit(X, y)

    area = st.number_input("Enter area (sq ft)")

    if st.button("Predict"):
        pred = model.predict([[area]])
        st.success(f"Estimated Price: ₹{pred[0]:,.2f} Lakhs")