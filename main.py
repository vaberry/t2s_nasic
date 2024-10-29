import streamlit as st
import os
from pathlib import Path
import sqlite3
import pandas as pd
from texttosql.sqlite import SQLiteEngine
import base64

# Set page configuration
st.set_page_config(
    page_title="NASIC",
    page_icon=":old_key:",
)

# Load and display the logo image
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{encoded_image}" width="500">
    </div>
    """.format(
        encoded_image=base64.b64encode(open("./nasic.png", "rb").read()).decode()
    ),
    unsafe_allow_html=True
)

# Initialize the engine at the global level
engine = None

def get_table_data(db_name, table_name, limit=1000):
    conn = sqlite3.connect(db_name)
    query = f"SELECT * FROM {table_name} LIMIT ?;"
    data = pd.read_sql(query, conn, params=(limit,))
    conn.close()
    return data

def display_query_result(query_input, db_name):
    with st.spinner("Processing your query..."):
        result = engine.query(query_input)

        generative_result = result.get('generative_result', 'No generative result found')
        sql = result.get('sql', 'No SQL found')
        sql_result = result.get('sql_result', 'No SQL result found')

        tab1, tab2, tab3 = st.tabs(["Answer", "SQL", "Data"])

        with tab1:
            st.write(generative_result)

        with tab2:
            st.code(sql, language='sql')

        with tab3:
            if isinstance(sql_result, list) and len(sql_result) > 0:
                try:
                    data = get_table_data(db_name)
                    if len(data) > 1000:
                        st.write("Displaying the first 1000 rows of pdw_data (subset of data):")
                    else:
                        st.write("Displaying the pdw_data:")
                    st.dataframe(data)
                except Exception as e:
                    st.json(sql_result)
                    st.error(f"Could not display table data: {str(e)}")
            else:
                st.json(sql_result)

def main():
    global engine

    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <h1>NASIC Weather Data</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload a file or multiple files from a directory",
        accept_multiple_files=True,
        type=["db", "csv", "json", "xlsx", "tmp"]
    )

    db_options = [f for f in os.listdir('.') if f.endswith('.db')]

    if db_options:
        selected_db = st.selectbox("Choose a database", db_options)

        if selected_db:
            engine = SQLiteEngine(db_name=f'{selected_db}')

        tab1, tab2 = st.tabs(["View Dataset", "Ask a Question"])

        with tab1:
            st.header("Database Viewer")
            data = get_table_data(selected_db, table_name=f'{selected_db[:-3]}')
            st.write("Displaying the first 1000 rows of data (subset of total data):")
            st.dataframe(data)
        
        with tab2:
            st.header("Database Chat")
            user_query = st.text_input("Ask a question:", "")

            if st.button("Submit"):
                if user_query:
                    st.write(f'{user_query}')
                    display_query_result(user_query, selected_db)
                else:
                    st.warning("Please enter a question...")
    else:
        st.warning("No databases found. Please upload a database file.")

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} file(s):")
        for uploaded_file in uploaded_files:
            db_name = Path(uploaded_file.name).stem
            engine = SQLiteEngine(db_name=db_name)
            tmp_dir = Path("uploaded_files")
            tmp_dir.mkdir(exist_ok=True)
            tmp_file_path = tmp_dir / uploaded_file.name
            with open(tmp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.endswith('.csv'):
                try:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        engine.create_tables_from_csv(tmp_file_path)
                    st.success(f"Successfully processed {uploaded_file.name} into database '{db_name}.db'")
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
            else:
                st.error("Please upload a CSV file to create tables.")

if __name__ == "__main__":
    main()
