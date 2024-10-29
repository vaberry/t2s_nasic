import sqlite3
import os
import pandas as pd
import json, re
from typing import Optional, Union, Dict, List
from pathlib import Path
import bluefile

class SQLiteDatabaseHandler:
    def __init__(self, db_name: str):
        self.db_name = Path(db_name).stem
        self.db_path = f"{self.db_name}.db"

        if not os.path.exists(self.db_path):
            print(f"Creating and connecting to new database '{self.db_path}' at the project root...")
            open(self.db_path, 'w').close()
        else:
            print(f"Connecting to existing database '{self.db_path}'...")

        try:
            with sqlite3.connect(self.db_path) as conn:
                print(f"Successfully connected to database '{self.db_path}'...")
        except sqlite3.Error as e:
            print(f"An error occurred while connecting to the database: {e}")

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def create_tables_from_tmp(self, tmp_path: Union[str, Path]):
        tmp_path = Path(tmp_path)
        with sqlite3.connect(self.db_path) as conn:
            if tmp_path.is_file() and tmp_path.suffix == '.tmp':
                self._import_tmp_to_db(conn, tmp_path)
            elif tmp_path.is_dir():
                for file in tmp_path.glob('*.tmp'):
                    self._import_tmp_to_db(conn, file)
            else:
                raise ValueError("The provided path must be a .tmp file or a directory containing .tmp files.")

    def _import_tmp_to_db(self, conn, tmp_file: Path):
        _, data = bluefile.read(tmp_file)
        c = conn.cursor()
        # Extract unique keys (field names) from the data
        fields = set()
        for entry in data:
            fields.update(entry.keys())

        # Create a table dynamically
        create_table_query = f"CREATE TABLE IF NOT EXISTS pdw_data ({', '.join(f'{field} TEXT' for field in fields)});"
        c.execute(create_table_query)

        # Insert data dynamically
        for entry in data:
            keys = ', '.join(entry.keys())
            placeholders = ', '.join('?' for _ in entry)
            values = tuple(str(entry[key]) for key in entry)
            insert_query = f"INSERT INTO pdw_data ({keys}) VALUES ({placeholders});"
            c.execute(insert_query, values)
        conn.commit()

    def create_tables_from_csv(self, csv_path: Union[str, Path]):
        with sqlite3.connect(self.db_path) as conn:
            csv_path = Path(csv_path)
            if csv_path.is_file() and csv_path.suffix == '.csv':
                self._import_csv_to_db(conn, csv_path)
            elif csv_path.is_dir():
                for file in csv_path.glob('*.csv'):
                    self._import_csv_to_db(conn, file)
            else:
                raise ValueError("The provided path must be a .csv file or a directory containing .csv files.")

    def _import_csv_to_db(self, conn: sqlite3.Connection, csv_file: Path):
        table_name = re.sub(r'\s+', '_', csv_file.stem.lower().strip())
        table_name = re.sub(r'[^a-zA-Z0-9_]', '', table_name)

        if self._table_exists(conn, table_name):
            print(f"Table '{table_name}' already exists in the database. Skipping import...")
        else:
            df = pd.read_csv(csv_file)
            df.columns = [re.sub(r'\s+', '_', col.lower().strip()) for col in df.columns]
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Table '{table_name}' created from file '{csv_file}'.")

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        return result is not None

    def get_db_schema(self) -> Optional[Dict[str, Dict[str, List[Dict[str, str]]]]]:
        if not os.path.exists(self.db_path):
            print(f"Database '{self.db_path}' does not exist.")
            return None

        schema = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get a list of all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    schema[table_name] = {
                        'columns': [],
                        'constraints': [],
                        'sample_data': []
                    }
                    
                    # Get the columns and their data types
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    for column in columns:
                        col_info = {
                            'name': column[1],
                            'type': column[2],
                            'primary_key': bool(column[5])
                        }
                        schema[table_name]['columns'].append(col_info)
                    
                    # Get table constraints (like unique, foreign keys, etc.)
                    cursor.execute(f"PRAGMA index_list({table_name});")
                    indexes = cursor.fetchall()
                    for index in indexes:
                        index_name = index[1]
                        cursor.execute(f"PRAGMA index_info({index_name});")
                        index_info = cursor.fetchall()
                        if index_info:
                            schema[table_name]['constraints'].append({
                                'index_name': index_name,
                                'columns': [info[2] for info in index_info],
                                'unique': bool(index[2])
                            })
                    
                    # Get foreign key constraints
                    cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                    foreign_keys = cursor.fetchall()
                    for fk in foreign_keys:
                        schema[table_name]['constraints'].append({
                            'type': 'foreign_key',
                            'from': fk[3],
                            'to': fk[4],
                            'table': fk[2]
                        })
                    
                    # Get the first 5 rows of data from the table
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                    rows = cursor.fetchall()
                    schema[table_name]['sample_data'] = rows

        except sqlite3.Error as e:
            print(f"An error occurred while retrieving the schema: {e}")
            return None
        
        return schema
    
    def update_primary_key(self, table_name: str, column_name: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if the table exists
            if not self._table_exists(conn, table_name):
                raise ValueError(f"Table '{table_name}' does not exist in the database.")

            # Check if the column exists in the table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_exists = any(column[1] == column_name for column in columns)
            
            if not column_exists:
                raise ValueError(f"Column '{column_name}' does not exist in table '{table_name}'.")

            # Create a new table with the primary key
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name}_temp AS SELECT * FROM {table_name};")
            cursor.execute(f"DROP TABLE {table_name};")
            
            column_definitions = ", ".join(
                f"{col[1]} {col[2]}" + (" PRIMARY KEY" if col[1] == column_name else "")
                for col in columns
            )
            
            cursor.execute(f"CREATE TABLE {table_name} ({column_definitions});")
            cursor.execute(f"INSERT INTO {table_name} SELECT * FROM {table_name}_temp;")
            cursor.execute(f"DROP TABLE {table_name}_temp;")
            
            conn.commit()
            print(f"Primary key added to column '{column_name}' in table '{table_name}'.")

    def remove_primary_key(self, table_name: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if the table exists
            if not self._table_exists(conn, table_name):
                raise ValueError(f"Table '{table_name}' does not exist in the database.")

            # Get the table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            # Check if there's a primary key to remove
            primary_keys = [col for col in columns if col[5] == 1]
            if not primary_keys:
                raise ValueError(f"Table '{table_name}' does not have a primary key.")

            # Create a new table without the primary key
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name}_temp AS SELECT * FROM {table_name};")
            cursor.execute(f"DROP TABLE {table_name};")

            column_definitions = ", ".join(
                f"{col[1]} {col[2]}"
                for col in columns
            )

            cursor.execute(f"CREATE TABLE {table_name} ({column_definitions});")
            cursor.execute(f"INSERT INTO {table_name} SELECT * FROM {table_name}_temp;")
            cursor.execute(f"DROP TABLE {table_name}_temp;")

            conn.commit()
            print(f"Primary key removed from table '{table_name}'.")

    def add_foreign_key(self, table_name: str, column_name: str, ref_table_name: str, ref_column_name: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if the table exists
            if not self._table_exists(conn, table_name):
                raise ValueError(f"Table '{table_name}' does not exist in the database.")

            # Check if the reference table exists
            if not self._table_exists(conn, ref_table_name):
                raise ValueError(f"Reference table '{ref_table_name}' does not exist in the database.")

            # Check if the column exists in the table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_exists = any(column[1] == column_name for column in columns)

            if not column_exists:
                raise ValueError(f"Column '{column_name}' does not exist in table '{table_name}'.")

            # Check if the reference column exists in the reference table and is a primary key
            cursor.execute(f"PRAGMA table_info({ref_table_name});")
            ref_columns = cursor.fetchall()
            ref_column_info = next((column for column in ref_columns if column[1] == ref_column_name), None)

            if ref_column_info is None:
                raise ValueError(f"Column '{ref_column_name}' does not exist in reference table '{ref_table_name}'.")

            if ref_column_info[5] != 1:
                raise ValueError(f"Column '{ref_column_name}' in table '{ref_table_name}' is not a primary key.")

            # Create a new table with the foreign key constraint
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name}_temp AS SELECT * FROM {table_name};")
            cursor.execute(f"DROP TABLE {table_name};")

            column_definitions = ", ".join(
                f"{col[1]} {col[2]}"
                for col in columns
            )

            foreign_key_constraint = f", FOREIGN KEY({column_name}) REFERENCES {ref_table_name}({ref_column_name})"

            create_table_query = f"CREATE TABLE {table_name} ({column_definitions}{foreign_key_constraint});"

            cursor.execute(create_table_query)
            cursor.execute(f"INSERT INTO {table_name} SELECT * FROM {table_name}_temp;")
            cursor.execute(f"DROP TABLE {table_name}_temp;")

            conn.commit()
            print(f"Foreign key added to column '{column_name}' in table '{table_name}', referencing '{ref_column_name}' in table '{ref_table_name}'.")

    def remove_foreign_key(self, table_name: str, column_name: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if the table exists
            if not self._table_exists(conn, table_name):
                raise ValueError(f"Table '{table_name}' does not exist in the database.")

            # Get the table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            # Check if the column exists in the table
            column_exists = any(column[1] == column_name for column in columns)
            if not column_exists:
                raise ValueError(f"Column '{column_name}' does not exist in table '{table_name}'.")

            # Get foreign keys for the table
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()

            # Check if the foreign key exists on the specified column
            foreign_key_info = next((fk for fk in foreign_keys if fk[3] == column_name), None)
            if not foreign_key_info:
                raise ValueError(f"No foreign key constraint found on column '{column_name}' in table '{table_name}'.")

            # Create a new table without the foreign key constraint
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name}_temp AS SELECT * FROM {table_name};")
            cursor.execute(f"DROP TABLE {table_name};")

            column_definitions = ", ".join(
                f"{col[1]} {col[2]}"
                for col in columns
            )

            create_table_query = f"CREATE TABLE {table_name} ({column_definitions});"

            cursor.execute(create_table_query)
            cursor.execute(f"INSERT INTO {table_name} SELECT * FROM {table_name}_temp;")
            cursor.execute(f"DROP TABLE {table_name}_temp;")

            conn.commit()
            print(f"Foreign key removed from column '{column_name}' in table '{table_name}'.")

    def execute_query(self, sql: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql)
                conn.commit()
                df = pd.read_sql_query(sql, conn)
                json_result = df.to_json(orient='records')
                print(json.dumps(json_result, indent=4))
                print("Query executed successfully...")
                return json_result
            except sqlite3.Error as e:
                print(f"An error occurred while executing the query: {e}")
                return None
