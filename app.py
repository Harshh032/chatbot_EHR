import streamlit as st
import pandas as pd
from PIL import Image
from main import Neo4jGPTQuery

# Configuration - replace with your actual credentials
NEO4J_URI = 'neo4j+s://4d08e27b.databases.neo4j.io'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = 'CZl9iRJn2OkIFbKnHP-st_5nsPSDCGCjaSBeWkQIs30'
GROQ_API_KEY = "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"

# Define the path to the logo file
LOGO_PATH = "MRC-white-tm.png"

# Function to load the logo image
def load_logo():
    try:
        return Image.open(LOGO_PATH)
    except FileNotFoundError:
        st.warning(f"Logo file not found at '{LOGO_PATH}'. Please check the file path.")
        return None

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gds_db" not in st.session_state:
    st.session_state.gds_db = Neo4jGPTQuery(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY)

# Load the logo
logo = load_logo()

# Sidebar for settings and info
with st.sidebar:
    # Display logo in sidebar
    if logo:
        st.image(logo, width=200, caption="", use_container_width=True)
    
    st.markdown('<p class="main-header">Healthcare Assistant</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("About")
    st.write("This assistant helps healthcare professionals query patient data using natural language.")
    
    st.markdown("---")
    
    show_cypher = st.toggle("Show Cypher Queries", value=False)
    show_raw_data = st.toggle("Show Raw Data", value=False)
    
    # Advanced options
    st.markdown("---")
    st.subheader("Advanced Options")
    entity_options = ["Auto (Try All)", "Immunization", "Medication", "Condition", "Observation", "Test", "AllergyIntolerance"]
    selected_entity = st.selectbox("Focus on entity type:", entity_options)
    
    st.markdown("---")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main UI
# Display logo in center of main area
if logo:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=300, caption="", use_container_width=True)

st.markdown('<h3 class="main-header">Healthcare Chat Assistant</h3>', unsafe_allow_html=True)
st.write("Ask questions about patient data in natural language.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            if "summary" in message:
                st.markdown(message["summary"])
                
                # Show Cypher query if enabled
                if show_cypher and "cypher" in message:
                    with st.expander("View Cypher Query"):
                        st.code(message["cypher"], language="cypher")
                
                # Show data table if enabled and results exist
                if show_raw_data and "results" in message and message["results"]:
                    with st.expander("View Raw Data"):
                        st.dataframe(pd.DataFrame(message["results"]))

# User input
if prompt := st.chat_input("Ask about patient data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show typing animation
        message_placeholder.markdown("⏳ Thinking...")
        
        # Determine if we should use a specific entity type
        entity_type = None if selected_entity == "Auto (Try All)" else selected_entity
        
        # Run the query
        if entity_type:
            # If specific entity type is selected, modify the query to focus on that type
            history = [{"role": "system", "content": f"Focus your query on {entity_type} entities."}]
            response = st.session_state.gds_db.run(prompt, history=history)
        else:
            # Otherwise run with the standard approach
            response = st.session_state.gds_db.run(prompt)
        
        if response["error"]:
            summary = f"I encountered an error with the database query: {response['error']}"
        elif not response["results"]:
            summary = "I couldn't find any data matching your query. Could you try rephrasing your question?"
        else:
            # Generate summary of results
            summary = st.session_state.gds_db.summarize_results(response["results"], prompt)
        
        # Update assistant message
        message_placeholder.markdown(summary)
        
        # Add response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "summary": summary,
            "cypher": response["cypher"],
            "results": response["results"]
        })

# Ensure the database connection is closed when the app is stopped
if "gds_db" in st.session_state and st.session_state.gds_db:
    import atexit
    atexit.register(st.session_state.gds_db.close)