import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
from groq import Groq

# from groq.errors import APIStatusError
from neo4j.exceptions import CypherSyntaxError
import time
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Healthcare Chat Assistant",
    page_icon="🏥",
    layout="centered"
)

# --- Custom CSS for ChatGPT-like UI ---
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .stButton>button {
        border-radius: 20px;
    }
    .query-box {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .result-table {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .center-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    .sidebar-logo {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Credentials ---
# Store these in Streamlit secrets in production
NEO4J_URI = "neo4j+s://4d08e27b.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "CZl9iRJn2OkIFbKnHP-st_5nsPSDCGCjaSBeWkQIs30"
GROQ_API_KEY = "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"

# --- Neo4j and Groq Integration ---
class Neo4jGPTQuery:
    def __init__(self, url, user, password, groq_api_key):
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        self.groq_client = Groq(api_key=groq_api_key)
        self.schema = self.generate_schema()

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def generate_schema(self):
        """Define the database schema structure for Cypher query generation."""
        return """
        Patient(id: STRING, name: STRING, gender: STRING, birthDate: STRING, telecom: STRING, address: STRING, maritalstatus: STRING)
        Condition(name: STRING, clinicalstatus: STRING, verificationstatus: STRING, severity: STRING, onsetdatetime: STRING, recordedDate: STRING)
        Observation(name: STRING, category: STRING, effectivedatetime: STRING, value: STRING, unit: STRING)
        Medication(name: STRING, status: STRING, dosage: STRING)
        Immunization(name: STRING, status: STRING, occurrenceDateTime: STRING, manufacturer: STRING)
        Test(name: STRING, status: STRING, effectiveDateTime: STRING)
        AllergyIntolerance(code: STRING, clinicalstatus: STRING, verificationstatus: STRING)

        Relationship types:
        (Patient)-[:HAS_CONDITION]->(Condition)
        (Patient)-[:HAS_OBSERVATION]->(Observation)
        (Patient)-[:TAKES]->(Medication)
        (Patient)-[:RECEIVED]->(Immunization)
        (Patient)-[:UNDERWENT]->(Test)
        (Patient)-[:HAS_ALLERGY]->(AllergyIntolerance)
        """

    def get_system_message(self, entity_type=None):
        """Generate system instructions for Cypher query generation."""
        schema_str = self.schema

        # If a specific entity type is provided, add guidance to focus on that entity
        entity_focus = ""
        if entity_type:
            entity_focus = f"\n- Focus your query on the {entity_type} entity type."

        return f"""
        Task: Generate a single Cypher query to query a Neo4j graph database based on the provided schema definition.
        Instructions:
        - Convert all queries to lowercase for consistency.
        - Use `CONTAINS` instead of `=` for flexible text searches.
        - Ensure all string comparisons use `toLower()`.
        - Return only ONE valid Cypher query without explanations.{entity_focus}

        Schema:
        {schema_str}

        Note: Do not include any explanations or apologies in your responses.
        """

    def get_summary_prompt(self, query, results):
        """Generate system instructions for summarizing query results."""
        return f"""
        Task: Summarize the following database query results in a clear, concise manner.

        Original query: {query}

        Results: {results}

        Provide a natural language summary that:
        1. Explains what information was found
        2. Highlights key data points and relationships
        3. Presents the information in an organized, easy-to-understand format
        4. Mentions if any data appears to be missing or incomplete
        """

    def construct_cypher(self, question, entity_type=None, history=None):
        """Generate a Cypher query using Groq API."""
        messages = [
            {"role": "system", "content": self.get_system_message(entity_type)},
            {"role": "user", "content": f"Generate a single Cypher query to find information about: {question}"},
        ]
        
        # Include conversation history if provided - but limit to recent messages
        if history:
            # Only use the last 5 message pairs to prevent token limit errors
            recent_history = history[-10:] if len(history) > 10 else history
            messages.extend(recent_history)

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                temperature=0.0,
                max_tokens=1000,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if e.status_code == 413:
                # If request was too large, try again without history
                if history:
                    try:
                        # Retry with just the system message and current question
                        response = self.groq_client.chat.completions.create(
                            model="llama3-70b-8192",
                            temperature=0.0,
                            max_tokens=1000,
                            messages=[
                                {"role": "system", "content": self.get_system_message(entity_type)},
                                {"role": "user", "content": f"Generate a single Cypher query to find information about: {question}"}
                            ]
                        )
                        return response.choices[0].message.content
                    except Exception as retry_error:
                        return f"Error generating query: {str(retry_error)}"
                else:
                    return f"Error: Request too large. Please try a shorter question."
            else:
                return f"API error: {str(e)}"
        except Exception as e:
            return f"Error generating query: {str(e)}"

    def summarize_results(self, question, results):
        """Summarize the query results using LLM."""
        # Convert results to a readable format
        if isinstance(results, str):  # Error message
            results_str = results
        else:
            # Format the results list into a more readable string
            results_str = "\n".join([str(record) for record in results])

        messages = [
            {"role": "system", "content": self.get_summary_prompt(question, results_str)},
            {"role": "user", "content": "Please summarize these database query results."}
        ]

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                temperature=0.3,
                max_tokens=1000,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if e.status_code == 413:
                # If the results are too large to summarize
                return "The results are extensive. Please refine your query for more specific information or view the raw data."
            else:
                return f"Error summarizing results: {str(e)}"
        except Exception as e:
            return f"Error summarizing results: {str(e)}"

    def query_database(self, cypher_query):
        """Execute the Cypher query on the Neo4j database."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                # Convert neo4j.Record objects to dictionaries
                return [record.data() for record in result]
        except CypherSyntaxError as e:
            return f"Invalid Cypher syntax: {str(e)}"
        except Exception as e:
            return f"Query error: {str(e)}"

    def run(self, question, entity_type=None, summarize=True, chat_history=None):
        """Handle the full process of generating and executing a Cypher query with optional summarization."""
        raw_results = None
        used_entity_type = None
        all_queries = {}  # Store all generated queries for display

        # Format chat history for LLM context
        formatted_history = []
        if chat_history:
            for msg in chat_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    # Only include user questions and assistant answers, not the full data
                    if msg["role"] == "user":
                        formatted_history.append({"role": "user", "content": msg["content"]})
                    elif msg["role"] == "assistant" and "content" in msg:
                        formatted_history.append({"role": "assistant", "content": msg["content"]})

        # If entity_type is not explicitly specified (or set to "Auto")
        if not entity_type or entity_type == "Auto":
            entity_types = ["Patient", "Immunization", "Medication", "Condition", "Observation", "AllergyIntolerance", "Test"]

            for ent_type in entity_types:
                cypher = self.construct_cypher(question, entity_type=ent_type, history=formatted_history)
                
                # Check if we got an error response
                if cypher.startswith("Error:") or cypher.startswith("API error:"):
                    all_queries[ent_type] = cypher
                    if "Request too large" in cypher:
                        # Early return with the error
                        return {
                            "question": question,
                            "entity_type": None,
                            "raw_results": None,
                            "summary": cypher + " The conversation history is too large. Please clear the chat and start a new conversation, or try a more specific question.",
                            "all_queries": all_queries,
                            "cypher": cypher
                        }
                    continue
                
                all_queries[ent_type] = cypher
                result = self.query_database(cypher)

                # If we got actual results (not an error string and not empty)
                if not isinstance(result, str) and result:
                    raw_results = result
                    used_entity_type = ent_type
                    break

            # If we've tried all entity types and found nothing, try without specifying
            if raw_results is None and not any(q.startswith("Error:") for q in all_queries.values()):
                cypher = self.construct_cypher(question, history=formatted_history)
                all_queries["General"] = cypher
                raw_results = self.query_database(cypher)
        else:
            # If entity_type is specified, just query for that type
            cypher = self.construct_cypher(question, entity_type=entity_type, history=formatted_history)
            
            # Check if we got an error response
            if cypher.startswith("Error:") or cypher.startswith("API error:"):
                all_queries[entity_type] = cypher
                if "Request too large" in cypher:
                    # Early return with the error
                    return {
                        "question": question,
                        "entity_type": None,
                        "raw_results": None,
                        "summary": cypher + " The conversation history is too large. Please clear the chat and start a new conversation, or try a more specific question.",
                        "all_queries": all_queries,
                        "cypher": cypher
                    }
            else:
                all_queries[entity_type] = cypher
                raw_results = self.query_database(cypher)
                used_entity_type = entity_type

        # Check if results are an error message
        is_error = isinstance(raw_results, str)
        
        # Generate a summary of the results using the LLM if requested
        if summarize and not is_error and raw_results:
            summary = self.summarize_results(question, raw_results)
        elif is_error:
            summary = raw_results  # Use the error message as summary
        else:
            summary = "No relevant data found. Try refining your query or selecting a different entity type."

        return {
            "question": question,
            "entity_type": used_entity_type,
            "raw_results": raw_results,
            "summary": summary,
            "all_queries": all_queries,
            "cypher": all_queries.get(used_entity_type if used_entity_type else "General", "")
        }

# --- Function to load logo ---
def load_logo():
    logo_path = "MRC-white-tm.png"
    try:
        return Image.open(logo_path)
    except FileNotFoundError:
        st.warning(f"Logo file not found at '{logo_path}'. Please check the file path.")
        return None

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "gds_db" not in st.session_state:
    st.session_state.gds_db = Neo4jGPTQuery(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY)

# --- Load logo ---
logo = load_logo()

# --- Sidebar for settings and info ---
with st.sidebar:
    # Display logo in sidebar
    if logo:
        st.image(logo, width=200, caption="", use_container_width=True)
    
    st.title("🏥 Healthcare Assistant")
    st.markdown("---")
    
    st.subheader("Query Settings")
    
    # Entity type selection
    entity_type = st.selectbox(
        "Entity Focus",
        ["Auto", "Patient", "Condition", "Observation", "Medication", 
         "Immunization", "Test", "AllergyIntolerance"],
        help="Select an entity type to focus the search, or use Auto to let the system decide"
    )
    
    st.markdown("---")
    
    # Display options
    st.subheader("Display Options")
  
    
    show_raw_data = st.toggle("Show Raw Data", value=False)
    show_entity_info = st.toggle("Show Entity Information", value=False)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    # About section
    st.markdown("---")
    st.subheader("About")
    st.write("This assistant helps healthcare professionals query patient data using natural language.")

# --- Main UI ---
# Display logo in center of main area
if logo:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=300, caption="", use_container_width=True)

st.title("Healthcare Chat Assistant")
st.write("Ask questions about patient data in natural language.")

# --- Display chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            if "summary" in message:
                st.markdown(message["summary"])
                
                # Show entity type if enabled
                if show_entity_info and "entity_type" in message and message["entity_type"]:
                    st.info(f"Entity Type: {message['entity_type']}")
                
    
                # Show data table if enabled and results exist
                if show_raw_data and "raw_results" in message and message["raw_results"]:
                    if not isinstance(message["raw_results"], str):  # Check if results are not an error message
                        with st.expander("View Raw Data"):
                            st.dataframe(pd.DataFrame(message["raw_results"]))

# --- User input ---
if prompt := st.chat_input("Ask about patient data..."):
    # Add user message to chat history
    prompt = prompt.lower()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show typing animation
        message_placeholder.markdown("⏳ Thinking...")
        
        # Process the entity type selection (use None if "Auto" is selected)
        selected_entity = None if entity_type == "Auto" else entity_type
        
        try:
            # Run the query with the selected entity type and chat history
            response = st.session_state.gds_db.run(
                question=prompt,
                entity_type=selected_entity,
                summarize=True,
                chat_history=st.session_state.chat_history
            )
            
            # Update assistant message with summary
            message_placeholder.markdown(response["summary"])
            
            # Prepare response to add to history
            assistant_response = {
                "role": "assistant", 
                "summary": response["summary"],
                "cypher": response.get("cypher", ""),
                "raw_results": response.get("raw_results", None),
                "entity_type": response.get("entity_type", None),
                "all_queries": response.get("all_queries", {})
            }
            
            # Add response to both displayed messages and chat history
            st.session_state.messages.append(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": response["summary"]})
            
        except Exception as e:
            # Handle any unexpected errors
            error_message = f"An error occurred: {str(e)}"
            message_placeholder.markdown(error_message)
            
            # Add error response to history
            st.session_state.messages.append({
                "role": "assistant", 
                "summary": error_message
            })
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})

# Close connection when the app closes
if st.session_state.gds_db:
    st.session_state.gds_db.close()
