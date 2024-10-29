from openai import OpenAI
from dotenv import load_dotenv
import json, re

load_dotenv(override=True)

client = OpenAI()

class SQLiteLLMHandler:
    def __init__(self):
        pass
    
    def make_texttosql_llm_call(self, query: str, schema: dict) -> dict:
        prompt = self._build_texttosql_llm_prompt(query, schema)
        
        if not prompt:
            print("Error: Generated prompt is empty or None.")
            return {}

        sql_generative_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant and a SQL expert for SQLite databases. Respond ONLY with a valid JSON object containing the specified keys and values, without any additional text, code blocks, or formatting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        
        result = sql_generative_response.choices[0].message.content.strip()
        result = re.sub(r"```(\w+)?", "", result).strip()
        
        # Convert true/false strings to Python booleans
        result = result.replace('true', 'True').replace('false', 'False')
        
        try:
            json_result = eval(result)
            if isinstance(json_result, dict):
                return json_result
            else:
                return {"error": "Response is not a dictionary", "response": result}
        except Exception as e:
            return {"error": f"Failed to decode JSON: {e}", "response": result}

    def _build_texttosql_llm_prompt(self, query: str, schema: dict) -> str:
        prompt = ''
        prompt += "You are a SQLite expert tasked with returning a SQL query based on a user's natural language question, the data scema, and a few example rows of the data."
        prompt += f"\n\nThe data schema is as follows:\n\n{schema}"
        prompt += f"\n\nThe user's question is: {query}"

        prompt += f"""
            The question will be run against a SQLite database.

            CONTEXTUAL INSTRUCTIONS:
            - Understand the context of the request to ensure the SQL query correctly identifies and filters for the relevant entities.
            - Maintain context from previous questions and ensure the current query builds on previous results when needed.
            - Use results from previous queries to inform and refine the current query.
            - Be mindful of pronouns and ambiguous terms, ensuring they are mapped to the correct entities and columns.

            SQL INSTRUCTIONS:
            - It is critical not to use parameterized queries.
            - Don't make up any new columns, only use the columns from the TABLE HEADs above.
            - When doing computation, only round to two decimal places. For example, 12.54321 should be 12.54. Try to always use floats instead of integers.
            - Don't use symbols when evaluating, like '?', '(', ')', '%', '$', ':', etc. For example, don't use things like 'SELECT key FROM popular_spotify_songs_2 WHERE LOWER(track_name) LIKE ?' or 'SELECT key FROM popular_spotify_songs_2 WHERE LOWER(track_name) LIKE (:name)'.
            - Don't use '=' to compare strings. Instead, use 'LIKE' for string comparisons.
            - Write a SQL query that retrieves data relevant to the query while adhering to general SQL standards. Ensure the query is adaptable to any dataset with the same schema.
            - Be mindful that the LLM's maximum context length is 128,000 tokens. Make sure the query won't bring back enough results to break that maximum context length.
            - Pay careful attention to the entities being discussed when creating the SQL query. For example, when asked about 'they', 'it', 'that', etc., ensure you are clear on the entity being referred to.
            - Only generate a single SQL statement. Avoid creating new columns or misaligning table columns.
            - Consider SQLite's specific limitations, as all queries will run on a SQLite database.
            - When filtering on date columns, format the dates as 'YYYY-MM-DD HH:MM:SS'. Ensure you're accounting for leap years and other date-related edge cases.
            - Implement case-insensitive comparisons. For example, use 'WHERE LOWER(pi_name) LIKE '%john%smith%' instead of 'WHERE pi_name LIKE '%John Smith%'.
            - Use the 'IN' operator for multiple fixed values instead of multiple 'LIKE' clauses. Example: 'WHERE pi_name IN ('john smith', 'jane doe')'.
            - Include wildcard placeholders to accommodate variations in data spacing, e.g., 'WHERE LOWER(pi_name) LIKE '%john%smith%' for 'John Smith'.
            - Optimize the query to return only the necessary data. Utilize SQL clauses like 'DISTINCT', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT', 'JOIN', 'UNION', etc., to refine the data retrieval.
            - In cases of long string comparisons involving multiple keywords, use 'OR' for non-stop-words to cover any of the conditions. Example: 'WHERE LOWER(text) LIKE '%anti-jamming%' OR LOWER(text) LIKE '%gps%' OR LOWER(text) LIKE '%navigation%'.
            - Aim to return comprehensive and relevant information by defaulting to broader, lowercase comparisons.

            LLM RESPONSE INSTRUCTIONS:
            - The LLM should first determine if the user's question is out of domain (i.e., unrelated to the content of the database).
            - If the question is out of domain, return the following key-value pairs:
                'out_of_domain': "Make this a Python True boolean value.",
                'out_of_domain_message': "Provide a message indicating that the query is out of domain.",
                'query_cleaning': "Cleaned version of the user's query, removing any extraneous or irrelevant parts.",
                'query_expansion': "Expanded version of the query, including any inferred or related details.",
                'recommended_next_questions': ["List a few relevant questions that might help the user understand the type of queries the database can handle."],
                'sql': "Make this an empty string."
            - If the question is in domain, return the following key-value pairs:
                'out_of_domain': "Make this a Python False boolean value.",
                'out_of_domain_message': "Make this an empty string.",
                'query_cleaning': "Cleaned version of the user's query, removing any extraneous or irrelevant parts.",
                'query_expansion': "Expanded version of the query, including any inferred or related details.",
                'recommended_next_questions': ["List of suggested follow-up questions based on the user's current query."],
                'sql': "Your SQL query here."

            - Return the key-value pairs as a Python dictionary. Do not include '```python' or '```' in the response.
            - Ensure the response returns a dictionary when evaluated: type(eval(response)) == dict.
            - The response should be a single dictionary with the specified keys and values.
            """
        
        return prompt
    
    def make_generative_llm_call(self, query: str, data: json) -> dict:
        prompt = self._build_generative_llm_prompt(query, data)

        if not prompt:
            print("Error: Generated prompt is empty or None.")
            return {}
        
        generative_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant and a SQL expert for SQLite databases. Respond ONLY with the answer to the user's question, without any additional text, code blocks, or formatting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        generative_result = generative_response.choices[0].message.content.strip()

        return generative_result
    
    def _build_generative_llm_prompt(self, query, data):
        return_prompt = f"""
        Based on the question:\n\n**{query}**\n\n the following data was found:\n\n
        SQL Data in JSON format: {data}\n

        Do not make reference the data sources themselves, only reference the data. For example, don't mention 'the data', 'JSON', 'SQL data', 'databases', etc.\n

        Unless specified in the query, don't show your work for mathematical calculations. Only provide the final answer.\n

        Unless specified otherwise, the answer should be generated based on the data provided in the prompt.\n

        Use organizing techniques like lists, bullet points, or paragraphs to structure your response.\n

        **This is a retrieval-augmented generation task, so it is critical that you only generate the answer based on the data provided in this prompt.**
        **If you need to make any assumptions, please state them clearly.**
        **If you think the data provided is insufficient to generate the answer, please state that as well.**
        """

        return return_prompt
