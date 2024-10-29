from texttosql.sqlite.handlers.database.handler import SQLiteDatabaseHandler
from texttosql.sqlite.handlers.query.handler import SQLiteQueryHandler
from texttosql.sqlite.handlers.llm.handler import SQLiteLLMHandler
import json

class SQLiteEngine(SQLiteDatabaseHandler, SQLiteQueryHandler, SQLiteLLMHandler):
    def __init__(self, db_name: str):
        # Initialize the SQLiteDatabaseHandler with the db_name
        SQLiteDatabaseHandler.__init__(self, db_name=db_name)
        
        # Initialize the SQLiteQueryHandler
        SQLiteQueryHandler.__init__(self)
        
        # Initialize the SQLiteLLMHandler
        SQLiteLLMHandler.__init__(self)

    def query(self, query: str):
        # Handle the query using the inherited handle_query method
        cleaned_query = self.handle_query(query)
        
        # Get the database schema using the inherited get_db_schema method
        schema = self.get_db_schema()
        
        # Make a text-to-SQL LLM call using the inherited make_texttosql_llm_call method
        llm_sql_result = self.make_texttosql_llm_call(query=cleaned_query, schema=schema)

        return_result = {}

        if isinstance(llm_sql_result, dict):
            if 'error' in llm_sql_result:
                return_result['sql_result'] = []
                return_result['generative_result'] = llm_sql_result.get('error')
                return return_result
            else:
                if llm_sql_result.get('out_of_domain'):
                    return_result['sql_result'] = []
                    out_of_domain_message = llm_sql_result.get('out_of_domain_message')
                    out_of_domain_message += "\n\nHere are some recommended questions:\n\n"
                    # add recommended questions
                    for i, question in enumerate(llm_sql_result.get('recommended_next_questions'), 1):
                        out_of_domain_message += f"{i}. {question}\n"
                    return_result['generative_result'] = out_of_domain_message
                    print(json.dumps(llm_sql_result, indent=4))
                    return return_result

        # Execute the query using the inherited execute_query method
        data = self.execute_query(llm_sql_result.get('sql'))
        
        if data:
            # Make a generative LLM call using the inherited make_generative_llm_call method
            llm_generative_result = self.make_generative_llm_call(query=cleaned_query, data=data)
            print(json.dumps(llm_generative_result, indent=4))
        
        for k, v in llm_sql_result.items():
            return_result[k] = v
        return_result['sql_result'] = data
        return_result['generative_result'] = llm_generative_result

        return return_result
