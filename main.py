import neo4j
import pandas as pd
from groq import Groq
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError

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
        Patient(id: STRING, name: STRING, gender: STRING, birthDate: STRING, telecom: STRING, address: STRING, maritalStatus: STRING)
        Condition(name: STRING, clinicalStatus: STRING, verificationStatus: STRING, severity: STRING, onsetDateTime: STRING, recordedDate: STRING)
        Observation(name: STRING, category: STRING, effectiveDateTime: STRING, value: STRING, unit: STRING)
        Medication(name: STRING, status: STRING, dosage: STRING)
        Immunization(name: STRING, status: STRING, occurrenceDateTime: STRING, manufacturer: STRING)
        Test(name: STRING, status: STRING, effectiveDateTime: STRING)
        AllergyIntolerance(code: STRING, clinicalStatus: STRING, verificationStatus: STRING)
        
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

    def construct_cypher(self, question, entity_type=None, history=None):
        """Generate a Cypher query using Groq API."""
        messages = [
            {"role": "system", "content": self.get_system_message(entity_type)},
            {"role": "user", "content": question},
        ]
        if history:
            messages.extend(history)

        response = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",  
            temperature=0.0,
            max_tokens=1000,
            messages=messages
        )
        return response.choices[0].message.content

    def query_database(self, cypher_query):
        """Execute the Cypher query on the Neo4j database."""
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]

    def run(self, question, history=None, retry=True):
        """Handle the full process of generating and executing a Cypher query."""
        
        # First try with all entity types
        all_results = []
        all_cyphers = {}
        error_message = None
        
        entity_types = ["Immunization", "Medication", "Condition", "Observation", "Test", "AllergyIntolerance"]
        
        # Try each entity type
        for entity_type in entity_types:
            try:
                cypher = self.construct_cypher(question, entity_type=entity_type, history=history)
                all_cyphers[entity_type] = cypher
                
                results = self.query_database(cypher)
                if results:
                    all_results.extend(results)
            except CypherSyntaxError as e:
                # Just continue to the next entity type if there's an error
                continue
            except Exception as e:
                error_message = str(e)
        
        # If we got no results from specific entity types, try a general query
        if not all_results and not error_message:
            try:
                cypher = self.construct_cypher(question, history=history)
                all_cyphers["general"] = cypher
                results = self.query_database(cypher)
                if results:
                    all_results.extend(results)
            except CypherSyntaxError as e:
                if not retry:
                    error_message = f"Invalid Cypher syntax: {str(e)}"
                else:
                    # Try to fix the query with error feedback
                    try:
                        retry_cypher = self.construct_cypher(
                            question,
                            history=[
                                {"role": "assistant", "content": cypher},
                                {
                                    "role": "user",
                                    "content": f"This query returns an error: {str(e)}. "
                                              f"Provide a corrected query that works without explanations or apologies.",
                                },
                            ]
                        )
                        all_cyphers["retry"] = retry_cypher
                        results = self.query_database(retry_cypher)
                        if results:
                            all_results.extend(results)
                    except Exception as retry_e:
                        error_message = str(retry_e)
            except Exception as e:
                error_message = str(e)
        
        # Combine all cyphers into a single string for display
        combined_cypher = "\n\n".join([f"--- {entity_type} Query ---\n{cypher}" for entity_type, cypher in all_cyphers.items()])
        
        return {
            "cypher": combined_cypher,
            "results": all_results,
            "error": error_message
        }

    def summarize_results(self, results, question):
        """Summarizes the query results using Groq."""
        if not results:
            return "No data found matching your query."

        messages = [
            {"role": "system", "content": "You are a helpful healthcare assistant. Summarize the following Neo4j patient data in a clear, conversational way. Focus on answering the user's original question."},
            {"role": "user", "content": f"Original question: {question}\n\nData: {str(results)}"}
        ]

        response = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=500,
            messages=messages
        )
        return response.choices[0].message.content