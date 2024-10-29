from typing import Optional

class SQLiteQueryHandler:
    def __init__(self):
        pass

    def handle_query(self, query: str):
        query = self._clean_query(query)
        return query

    def _clean_query(self, query: str) -> str:
        return query