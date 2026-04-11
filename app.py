import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. App Configuration
st.set_page_config(page_title="Macroeconomic AI Agent", page_icon="🌍", layout="wide")
st.title("🌍 Crisis Management Agentic System")
st.markdown("""
This application acts as an **AI-augmented Macroeconomic Risk Dashboard**. 
Select a country and watch the AI run a **Multi-Prompt Orchestration**, analyzing the data through 3 different cognitive frameworks, followed by a Meta-Analysis of its own reasoning.
""")

# 2. Data Caching
@st.cache_data
def load_data():
    data_path = Path("data/processed_macro_risk.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

df = load_data()

if df is None:
    st.error("❌ Data not found. Please verify the path.")
    st.stop()

# 3. User Interface (UI)
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Global Risk Matrix")
    risk_colors = {'Critical': '#d62828', 'Severe': '#f77f00', 'High': '#fcbf49', 'Moderate': '#2a9d8f', 'Low': '#003049', 'Extreme': '#000000'}
    fig = px.scatter(
        df, x='pct_increase', y='gdp_impact_pct', size='population_m', color='vulnerability',
        text='country', hover_name='country', color_discrete_map=risk_colors, height=550
    )
    fig.update_traces(textposition='top center')
    fig.add_hline(y=-2.0, line_dash="dash", line_color="gray", annotation_text="Severe Recession")
    fig.add_vline(x=10.0, line_dash="dash", line_color="gray", annotation_text="High Price Shock")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🤖 Multi-Prompt AI Orchestration")
    country_list = df['country'].tolist()
    selected_country = st.selectbox("Select a country to analyze:", country_list)
    
    country_data = df[df['country'] == selected_country].iloc[0]
    
    # Key Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Fuel Surge", value=f"{country_data['pct_increase']:.2f}%")
    m2.metric(label="GDP Impact", value=f"{country_data['gdp_impact_pct']:.2f}%")
    m3.metric(label="Risk Level", value=country_data['vulnerability'])
    
    # --- 🧠 PROMPT DEFINITIONS ---
    
    prompt_zero_shot = """You are a Macroeconomic Analyst. Data: Country: {country}, Fuel Surge: {pct_increase}%, GDP Impact: {gdp_impact_pct}%, Risk: {vulnerability}. Task: Write a concise, 2-sentence summary of this country's risk."""
    
    prompt_few_shot = """You are a Macroeconomic Analyst. Output the assessment EXACTLY in this format.
    Example Input: Country: FakeLand, Fuel Surge: 15.0%, GDP Impact: -3.0%, Risk: High
    Example Output:
    * **Primary Threat:** FakeLand faces a High risk level due to a 15.0% fuel price surge.
    * **Macro Impact:** This shock directly triggers a GDP contraction of -3.0%.

    Real Input: Country: {country}, Fuel Surge: {pct_increase}%, GDP Impact: {gdp_impact_pct}%, Risk: {vulnerability}.
    Real Output:"""
    
    prompt_cot = """You are a Senior Economist. Analyze the economic shock for {country} (Fuel Surge: {pct_increase}%, GDP Impact: {gdp_impact_pct}%, Risk: {vulnerability}).
    Think step-by-step:
    1. First, analyze the consumer impact of the {pct_increase}% fuel surge.
    2. Second, correlate that to the {gdp_impact_pct}% projected GDP change.
    3. Conclude why this is categorized as '{vulnerability}'.
    Format your response with clear numbered steps."""

    # META-PROMPT: La IA se analiza a sí misma
    prompt_meta = """You are an Expert AI Prompt Engineer. I used three different prompt techniques to make an LLM analyze {country}'s economic risk.
    
    Zero-Shot Output: {zero_res}
    Few-Shot Output: {few_res}
    Chain-of-Thought Output: {cot_res}
    
    Task: In one concise paragraph, explain to a non-technical manager the differences between these three outputs. Explain how the 'Few-Shot' output is more structured than 'Zero-Shot', and how the 'Chain-of-Thought' reveals the logical reasoning behind the numbers."""

    # Generate Button
    if st.button("🚀 Run AI Orchestration Engine", type="primary"):
        with st.spinner("🧠 Orchestrating 4 sequential AI inferences with Llama 3. Please wait (~10-20 seconds)..."):
            try:
                llm = ChatOllama(model="llama3.2:1b", temperature=0.1)
                parser = StrOutputParser()
                
                inputs = {
                    "country": country_data['country'], "pct_increase": country_data['pct_increase'],
                    "gdp_impact_pct": country_data['gdp_impact_pct'], "vulnerability": country_data['vulnerability']
                }
                
                # 1. Ejecutar Zero-Shot
                zero_res = (PromptTemplate.from_template(prompt_zero_shot) | llm | parser).invoke(inputs)
                
                # 2. Ejecutar Few-Shot
                few_res = (PromptTemplate.from_template(prompt_few_shot) | llm | parser).invoke(inputs)
                
                # 3. Ejecutar Chain-of-Thought
                cot_res = (PromptTemplate.from_template(prompt_cot) | llm | parser).invoke(inputs)
                
                # 4. Ejecutar Meta-Analysis (pasándole los 3 resultados anteriores)
                meta_res = (PromptTemplate.from_template(prompt_meta) | llm | parser).invoke({
                    "country": inputs["country"],
                    "zero_res": zero_res, "few_res": few_res, "cot_res": cot_res
                })
                
                # --- VISUALIZACIÓN EN PESTAÑAS (TABS) ---
                st.success("✅ AI Orchestration Complete!")
                
                tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Zero-Shot", "2️⃣ Few-Shot", "3️⃣ Chain-of-Thought", "🧠 Meta-Analysis (AI on AI)"])
                
                with tab1:
                    st.info("**Zero-Shot (Direct Instruction):**\n\n" + zero_res)
                with tab2:
                    st.success("**Few-Shot (Structured Output):**\n\n" + few_res)
                with tab3:
                    st.warning("**Chain-of-Thought (Step-by-Step Reasoning):**\n\n" + cot_res)
                with tab4:
                    st.error("**🤖 Prompt Engineering Meta-Analysis:**\n\n" + meta_res)
                
            except Exception as e:
                st.error(f"Error connecting to Ollama: {e}. Is Ollama running?")