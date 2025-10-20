import streamlit as st

from vanna.remote import VannaDefault
import psycopg2

from sqlalchemy import create_engine, text

def table_exists(engine, table_name):
    """Check if table exists using SQLAlchemy engine"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = :table_name
                );
            """), {"table_name": table_name})
            exists = result.fetchone()[0]
            return exists
    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False

def insert_csv_to_database(df, table_name):
    # Create SQLAlchemy connection string
    connection_string = (
        f"postgresql://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}"
        f"@{st.secrets['DB_HOST']}:{st.secrets['DB_PORT']}/{st.secrets['DB_NAME']}"
    )
    
    engine = None
    try:
        # Create SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Check if table already exists
        if table_exists(engine, table_name):
            # Table exists, so drop it and recreate with new schema
            with engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            
            # Insert new data (this will create the table with the correct schema)
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists='replace',
                index=False
            )
            
            rows_inserted = len(df)
            
            return {
                "status": "success",
                "message": f"Table '{table_name}' replaced and {rows_inserted} new rows inserted.",
                "rows_inserted": rows_inserted
            }
        else:
            # Table doesn't exist, so create it and insert the data
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists='replace',
                index=False
            )
            
            rows_inserted = len(df)
            
            return {
                "status": "success",
                "message": f"Table '{table_name}' created and {rows_inserted} rows inserted successfully.",
                "rows_inserted": rows_inserted
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error inserting data: {str(e)}",
            "rows_inserted": 0
        }
    
    finally:
        # Always dispose of the engine
        if engine is not None:
            engine.dispose()

@st.cache_resource(ttl=3600)
def setup_vanna():
    vn = VannaDefault(api_key=st.secrets.get("VANNA_API_KEY"), model='artman-jr')
    vn.connect_to_postgres(
        host=st.secrets["DB_HOST"],
        dbname=st.secrets["DB_NAME"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        port=st.secrets["DB_PORT"]
    )
    return vn

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
    """Drop a table if it exists in the database"""

    
    connection_string = (
        f"postgresql://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}"
        f"@{st.secrets['DB_HOST']}:{st.secrets['DB_PORT']}/{st.secrets['DB_NAME']}"
    )
    
    engine = None
    try:
        engine = create_engine(connection_string)
        
        with engine.begin() as connection:
            # Drop table if exists
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
        
        return {
            "status": "success",
            "message": f"Table '{table_name}' dropped successfully (if it existed)"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error dropping table: {str(e)}"
        }
    
    finally:
        if engine is not None:
            engine.dispose()