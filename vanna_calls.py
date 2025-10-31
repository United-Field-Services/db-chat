import streamlit as st
from vanna.remote import VannaDefault
import psycopg2
from sqlalchemy import create_engine, text
from supabase import create_client, Client

# ============================================
# SUPABASE REST API CLIENT (for data insertion)
# ============================================

@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize Supabase client for REST API operations"""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def create_table_from_dataframe(df, table_name):
    """
    Create/recreate table in Supabase using direct psycopg2 connection
    """
    import psycopg2
    
    try:
        # Type mapping for PostgreSQL
        type_mapping = {
            'int64': 'INTEGER',
            'float64': 'REAL',
            'object': 'TEXT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP'
        }
        
        # Build column definitions
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sql_type = type_mapping.get(dtype, 'TEXT')
            col_lower = col.lower()
            columns.append(f"{col_lower} {sql_type}")
        
        # Build connection string
        conn_string = (
            f"host={st.secrets['DB_HOST']} "
            f"dbname={st.secrets['DB_NAME']} "
            f"user={st.secrets['DB_USER']} "
            f"password={st.secrets['DB_PASSWORD']} "
            f"port={st.secrets['DB_PORT']}"
        )
        
        # Connect and execute DDL
        conn = psycopg2.connect(conn_string)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Drop existing table and create new one
        drop_sql = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
        create_sql = f"CREATE TABLE {table_name} ({', '.join(columns)});"
        
        cursor.execute(drop_sql)
        cursor.execute(create_sql)
        
        cursor.close()
        conn.close()
        
        return {"status": "success", "message": f"Table '{table_name}' created successfully"}
        
    except Exception as e:
        return {"status": "error", "message": f"Error creating table: {str(e)}"}


def insert_csv_to_database(df, table_name):
    """
    Insert DataFrame to Supabase using REST API
    This works reliably on Streamlit Cloud (uses HTTPS)
    """
    try:
        # First, create/recreate the table with correct schema
        create_result = create_table_from_dataframe(df, table_name)
        if create_result["status"] != "success":
            return create_result
        
        # Clean the data: replace NaN with None (null in JSON)
        df_clean = df.replace({float('nan'): None})
        
        # Also handle inf values if any
        import numpy as np
        df_clean = df_clean.replace([np.inf, -np.inf], None)
        
        # Convert DataFrame to list of dictionaries
        data = df_clean.to_dict('records')
        
        # Now insert data using Supabase REST API
        supabase = get_supabase_client()
        
        # Insert new data in batches (Supabase has a limit per request)
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            response = supabase.table(table_name).upsert(batch).execute()
            total_inserted += len(batch)
        
        return {
            "status": "success",
            "message": f"Table recreated and {total_inserted} rows inserted into '{table_name}'",
            "rows_inserted": total_inserted
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error inserting data: {str(e)}",
            "rows_inserted": 0
        }


# ============================================
# VANNA SETUP (for SQL generation and queries)
# ============================================

@st.cache_resource(ttl=3600)
def setup_vanna():
    """Setup Vanna with PostgreSQL connection for querying"""
    vn = VannaDefault(api_key=st.secrets.get("VANNA_API_KEY"), model='artman-jr')
    
    # Vanna uses direct PostgreSQL connection for running queries
    # This should work because Vanna runs queries, not Streamlit Cloud
    vn.connect_to_postgres(
        host=st.secrets["DB_HOST"],
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        port=st.secrets["DB_PORT"]
    )
    return vn


# ============================================
# VANNA CACHED FUNCTIONS
# ============================================

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    return vn.generate_sql(question=question, allow_llm_to_see_data=True)


@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)


@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    return vn.run_sql(sql=sql)


@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)


@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)


@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)


def remove_training_data(vn):
    """Remove all training data from the model"""
    
    try:
        # Get all training data
        training_data = vn.get_training_data()
        
        # Handle case where training_data is None or empty
        if training_data is None or (hasattr(training_data, 'empty') and training_data.empty):
            return {
                "status": "success", 
                "message": "No training data to remove",
                "removed_count": 0
            }
        
        # Convert DataFrame to list of dicts if it's a DataFrame
        if hasattr(training_data, 'to_dict'):
            # It's a DataFrame
            records = training_data.to_dict('records')
            total_entries = len(records)
        else:
            return {
                "status": "error",
                "message": f"Unexpected training data format: {type(training_data)}",
                "removed_count": 0
            }
        
        if total_entries == 0:
            return {
                "status": "success", 
                "message": "No training data to remove",
                "removed_count": 0
            }
        
        # Remove each training data entry by ID
        removed_count = 0
        failed_count = 0
        
        for item in records:
            if 'id' in item and item['id'] is not None:
                try:
                    vn.remove_training_data(id=item['id'])
                    removed_count += 1
                except Exception as remove_error:
                    failed_count += 1
                    print(f"Failed to remove training data ID {item['id']}: {str(remove_error)}")
        
        # Verify removal
        try:
            remaining_data = vn.get_training_data()
            if remaining_data is not None and hasattr(remaining_data, '__len__'):
                remaining_count = len(remaining_data)
            else:
                remaining_count = 0
        except:
            remaining_count = 0
        
        if remaining_count > 0:
            return {
                "status": "warning",
                "message": f"Removed {removed_count} entries, but {remaining_count} still remain (failed: {failed_count})",
                "removed_count": removed_count,
                "remaining_count": remaining_count
            }
        
        message = f"Successfully removed all {removed_count} training data entries"
        if failed_count > 0:
            message += f" ({failed_count} failed)"
        
        return {
            "status": "success", 
            "message": message,
            "removed_count": removed_count
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        return {
            "status": "error",
            "message": f"Error removing training data: {str(e)}",
            "removed_count": 0
        }


def drop_table_if_exists(table_name):
    """Drop a table using Supabase REST API"""
    try:
        supabase = get_supabase_client()
        
        # Delete all rows from the table
        supabase.table(table_name).delete().neq('id', -999999).execute()
        
        return {
            "status": "success",
            "message": f"Table '{table_name}' cleared successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error clearing table: {str(e)}"
        }