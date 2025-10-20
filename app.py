import time
import streamlit as st
import pandas as pd

# ============================================
# YOUR APP CODE STARTS HERE
# No password protection needed - handled by Streamlit Cloud!
# ============================================

from vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached,
    setup_vanna,
    insert_csv_to_database,
    remove_training_data
)

avatar_url = "https://vanna.ai/img/vanna.svg"

st.set_page_config(layout="wide")

# Hardcoded table name
TABLE_NAME = "test"


def generate_ddl_from_dataframe(df, table_name="test"):
    """Automatically generate CREATE TABLE statement from DataFrame"""
    type_mapping = {
        'int64': 'INTEGER',
        'float64': 'REAL',
        'object': 'TEXT',
        'bool': 'INTEGER',
        'datetime64[ns]': 'DATETIME'
    }
    
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sql_type = type_mapping.get(dtype, 'TEXT')
        # Convert column name to lowercase for PostgreSQL compatibility
        col_lower = col.lower()
        columns.append(f"    {col_lower} {sql_type}")
    
    ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(columns) + "\n);"
    return ddl


def generate_documentation_from_dataframe(df, table_name="test"):
    """Automatically generate documentation from DataFrame"""
    num_rows = len(df)
    num_cols = len(df.columns)
    
    doc = f"Table: {table_name}\n"
    doc += f"Description: This table contains {num_rows} rows and {num_cols} columns.\n\n"
    doc += "Columns:\n"
    
    for col in df.columns:
        # Use lowercase column names for PostgreSQL compatibility
        col_lower = col.lower()
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        
        # Get sample unique values for categorical columns
        if dtype == 'object' and df[col].nunique() < 20:
            unique_vals = df[col].unique()[:5]
            doc += f"- {col_lower} ({dtype}): {non_null}/{num_rows} non-null values. "
            doc += f"Sample values: {', '.join(map(str, unique_vals))}\n"
        elif dtype in ['int64', 'float64']:
            min_val = df[col].min()
            max_val = df[col].max()
            doc += f"- {col_lower} ({dtype}): {non_null}/{num_rows} non-null values. "
            doc += f"Range: {min_val} to {max_val}\n"
        else:
            doc += f"- {col_lower} ({dtype}): {non_null}/{num_rows} non-null values\n"
    
    return doc


def generate_sample_questions(df, table_name="test"):
    """Generate sample questions based on DataFrame columns"""
    questions = []
    # Convert all column names to lowercase for PostgreSQL compatibility
    cols = [col.lower() for col in df.columns.tolist()]
    
    # Numeric columns (use lowercase)
    numeric_cols = [col.lower() for col in df.select_dtypes(include=['int64', 'float64']).columns.tolist()]
    
    # Categorical columns (use lowercase)
    categorical_cols = [col.lower() for col in df.select_dtypes(include=['object']).columns.tolist()]
    
    # Generate questions
    if len(cols) > 0:
        questions.append(f"What are all the columns in {table_name}?")
        questions.append(f"Show me the first 10 rows from {table_name}")
    
    if len(numeric_cols) > 0:
        questions.append(f"What is the average {numeric_cols[0]}?")
        if len(numeric_cols) > 1:
            questions.append(f"What is the sum of {numeric_cols[0]} grouped by {cols[1] if len(cols) > 1 else numeric_cols[1]}?")
    
    if len(categorical_cols) > 0:
        questions.append(f"How many unique {categorical_cols[0]} are there?")
        if len(categorical_cols) > 1:
            questions.append(f"Show me the count of records by {categorical_cols[0]}")
    
    if len(cols) >= 2:
        questions.append(f"Show me {cols[0]} and {cols[1]} from {table_name}")
    
    return questions[:5]  # Return max 5 questions


# Sidebar - Training Data Management
st.sidebar.title("🎓 Training Data")

# Upload CSV for training
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV for Training",
    type=['csv'],
    help="System will auto-generate schema & documentation"
)

if uploaded_file is not None:
    try:
        df_training = pd.read_csv(uploaded_file)
        
        # Normalize DataFrame column names for PostgreSQL compatibility
        # 1. Convert to lowercase
        # 2. Replace hyphens with underscores
        # 3. Replace spaces with underscores
        # 4. Remove other special characters
        df_training.columns = (
            df_training.columns
            .str.lower()
            .str.replace('-', '_', regex=False)
            .str.replace(' ', '_', regex=False)
            .str.replace(r'[^\w]', '_', regex=True)  # Replace any non-alphanumeric with underscore
            .str.replace(r'_+', '_', regex=True)  # Replace multiple underscores with single
            .str.strip('_')  # Remove leading/trailing underscores
        )
        
        # Show preview
        st.sidebar.success(f"✅ Loaded: {len(df_training)} rows × {len(df_training.columns)} columns")
        
        # Show preview toggle
        show_preview = st.sidebar.checkbox("Show Preview", value=False)
        
        if show_preview:
            st.sidebar.dataframe(df_training.head(3), use_container_width=True)
        
        # Train button
        if st.sidebar.button("🚀 Train Model", use_container_width=True, type="primary"):
            vn = setup_vanna()
            
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            # Step 0: Reset training data
            status_text.text("Step 0/5: Resetting previous training...")
            
            reset_result = remove_training_data(vn)
            
            if reset_result["status"] == "success":
                st.sidebar.success(f"✅ {reset_result['message']}")
            else:
                st.sidebar.warning(f"⚠️ {reset_result['message']}")
            
            progress_bar.progress(0.1)
            time.sleep(0.5)
    
            # Step 1: Insert data to database (will replace if exists)
            status_text.text("Step 1/4: Inserting data...")
            result = insert_csv_to_database(df_training, TABLE_NAME)
            
            if result["status"] == "success":
                st.sidebar.success(f"✅ {result['message']}")
            else:
                st.sidebar.error(f"❌ {result['message']}")
                st.stop()
            
            progress_bar.progress(0.3)
    
            # Step 2: DDL
            status_text.text("Step 2/4: Adding schema...")
            ddl = generate_ddl_from_dataframe(df_training, TABLE_NAME)
            vn.train(ddl=ddl)
            progress_bar.progress(0.5)
            
            # Step 3: Documentation
            status_text.text("Step 3/4: Adding documentation...")
            documentation = generate_documentation_from_dataframe(df_training, TABLE_NAME)
            vn.train(documentation=documentation)
            progress_bar.progress(0.7)
            
            # Step 4: Sample Q&A
            status_text.text("Step 4/4: Training Q&A pairs...")
            sample_questions = generate_sample_questions(df_training, TABLE_NAME)
            
            for idx, question in enumerate(sample_questions):
                if "first 10 rows" in question.lower():
                    sql = f"SELECT * FROM {TABLE_NAME} LIMIT 10"
                elif "average" in question.lower():
                    numeric_col = df_training.select_dtypes(include=['int64', 'float64']).columns[0].lower()
                    sql = f"SELECT AVG({numeric_col}) FROM {TABLE_NAME}"
                elif "count" in question.lower():
                    categorical_col = df_training.select_dtypes(include=['object']).columns[0].lower()
                    sql = f"SELECT {categorical_col}, COUNT(*) FROM {TABLE_NAME} GROUP BY {categorical_col}"
                else:
                    sql = f"SELECT * FROM {TABLE_NAME}"
                
                vn.train(question=question, sql=sql)
                progress = 0.7 + (0.2 * (idx + 1) / len(sample_questions))
                progress_bar.progress(progress)
            
            # Add PostgreSQL-specific training examples
            status_text.text("Step 4/4: Adding PostgreSQL-specific examples...")
            
            # Get column names
            categorical_cols = [col.lower() for col in df_training.select_dtypes(include=['object']).columns.tolist()]
            numeric_cols = [col.lower() for col in df_training.select_dtypes(include=['int64', 'float64']).columns.tolist()]
            all_cols = df_training.columns.tolist()
            
            # PostgreSQL COUNT with conditions examples
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                # Example: conditional counting using SUM with CASE
                vn.train(
                    question=f"Count records by different {cat_col} values",
                    sql=f"SELECT {cat_col}, COUNT(*) AS count FROM {TABLE_NAME} GROUP BY {cat_col}"
                )
                
                # Example: conditional sum with CASE WHEN
                vn.train(
                    question=f"Count how many records have {num_col} greater than 1",
                    sql=f"SELECT SUM(CASE WHEN {num_col} > 1 THEN 1 ELSE 0 END) AS count FROM {TABLE_NAME}"
                )
            
            # Add general feature analysis patterns
            # Try to identify target/outcome columns
            target_cols = [col for col in df_training.columns.str.lower() 
                          if any(keyword in col for keyword in ['cancer', 'disease', 'outcome', 'diagnosis', 'result', 
                                                                 'target', 'label', 'class', 'status', 'response'])]
            
            if len(target_cols) > 0:
                target_col = target_cols[0]
                
                # Get unique values in target column to determine positive class
                unique_vals = df_training[target_col].dropna().unique()
                positive_values = []
                
                # Heuristic to find positive values
                for val in unique_vals:
                    val_str = str(val).upper()
                    if val_str in ['YES', 'TRUE', '1', 'POSITIVE', 'HIGH']:
                        positive_values.append(str(val))
                
                # If we found positive values, create training examples
                if len(positive_values) > 0:
                    positive_val = positive_values[0]
                    
                    # Get feature columns (exclude target and non-numeric)
                    feature_cols = [col for col in all_cols if col.lower() != target_col]
                    numeric_features = [col for col in numeric_cols if col != target_col]
                    
                    # Example 1: Simple count with filter
                    if len(numeric_features) >= 1:
                        feat1 = numeric_features[0]
                        vn.train(
                            question=f"How many records have {feat1} equal to 2 among positive {target_col} cases?",
                            sql=f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {target_col} = '{positive_val}' AND {feat1} = 2"
                        )
                    
                    # Example 2: Ratio calculation with FILTER (only if 2+ features)
                    if len(numeric_features) >= 2:
                        feat1, feat2 = numeric_features[0], numeric_features[1]
                        vn.train(
                            question=f"What is the ratio of records with both {feat1} and {feat2} among positive {target_col}?",
                            sql=f"SELECT (COUNT(*) FILTER (WHERE {feat1} = 2 AND {feat2} = 2) * 1.0 / NULLIF(COUNT(*), 0)) AS ratio FROM {TABLE_NAME} WHERE {target_col} = '{positive_val}'"
                        )
                    
                    # Example 3: Rank features by frequency (limit to 5 features max)
                    if len(numeric_features) >= 3:
                        features_to_rank = numeric_features[:5]  # Limit to 5 features
                        union_parts = []
                        for feat in features_to_rank:
                            union_parts.append(f"SELECT '{feat}' AS feature, COUNT(*) AS frequency FROM {TABLE_NAME} WHERE {target_col} = '{positive_val}' AND {feat} = 2")
                        
                        ranking_sql = "\nUNION ALL\n".join(union_parts) + "\nORDER BY frequency DESC"
                        
                        vn.train(
                            question=f"Rank features by frequency with positive {target_col}",
                            sql=ranking_sql
                        )
                    
                    # Example 4: Count by feature value
                    if len(numeric_features) > 0:
                        sample_feature = numeric_features[0]
                        vn.train(
                            question=f"Count positive {target_col} cases grouped by {sample_feature}",
                            sql=f"SELECT {sample_feature}, COUNT(*) AS count FROM {TABLE_NAME} WHERE {target_col} = '{positive_val}' GROUP BY {sample_feature} ORDER BY count DESC"
                        )
                    
                    # Example 5: Feature comparison across target values
                    if len(numeric_features) > 0:
                        sample_feature = numeric_features[0]
                        vn.train(
                            question=f"Compare {sample_feature} values across {target_col} categories",
                            sql=f"SELECT {target_col}, {sample_feature}, COUNT(*) AS count FROM {TABLE_NAME} GROUP BY {target_col}, {sample_feature} ORDER BY {target_col}, {sample_feature}"
                        )
            
            # Add general aggregation patterns regardless of target column
            if len(numeric_cols) >= 2:
                num1, num2 = numeric_cols[0], numeric_cols[1]
                
                # Correlation/co-occurrence pattern
                vn.train(
                    question=f"Show frequency of {num1} and {num2} value combinations",
                    sql=f"SELECT {num1}, {num2}, COUNT(*) AS count FROM {TABLE_NAME} GROUP BY {num1}, {num2} ORDER BY count DESC"
                )
            
            progress_bar.progress(1.0)
            
            status_text.text("✅ Training complete!")
            
            st.sidebar.success(f"🎉 Successfully trained with:\n- Schema\n- Documentation\n- {len(sample_questions)} Q&A pairs")
            time.sleep(3)
            st.rerun()
            
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

st.sidebar.divider()

# Manual Q&A Training
st.sidebar.subheader("Manual Q&A")
question_input = st.sidebar.text_input("Question", placeholder="What is total sales?")
sql_input = st.sidebar.text_area("SQL Query", height=80, placeholder="SELECT SUM(sales) FROM orders")

if st.sidebar.button("Add Q&A", use_container_width=True):
    if question_input.strip() and sql_input.strip():
        vn = setup_vanna()
        vn.train(question=question_input, sql=sql_input)
        st.sidebar.success("✅ Q&A added!")
    else:
        st.sidebar.warning("⚠️ Enter both fields")

st.sidebar.divider()

# Output Settings
st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=False, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Summary", value=True, key="show_summary")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")

st.sidebar.divider()
st.sidebar.button("🔄 Reset Chat", on_click=lambda: st.session_state.update({"my_question": None}), use_container_width=True)

# Main title
st.title("🤖 Vanna AI - Natural Language Database Query")
st.caption("Ask questions about your data in plain English")


def set_question(question):
    st.session_state["my_question"] = question


assistant_message_suggested = st.chat_message(
    "assistant", avatar=avatar_url
)
if assistant_message_suggested.button("💡 Click to show suggested questions"):
    st.session_state["my_question"] = None
    questions = generate_questions_cached()
    for i, question in enumerate(questions):
        time.sleep(0.05)
        button = st.button(
            question,
            on_click=set_question,
            args=(question,),
        )

my_question = st.session_state.get("my_question", default=None)

if my_question is None:
    my_question = st.chat_input(
        "Ask me a question about your data",
    )


if my_question:
    st.session_state["my_question"] = my_question
    user_message = st.chat_message("user")
    user_message.write(f"{my_question}")

    sql = generate_sql_cached(question=my_question)

    if sql:
        if is_sql_valid_cached(sql=sql):
            if st.session_state.get("show_sql", True):
                assistant_message_sql = st.chat_message(
                    "assistant", avatar=avatar_url
                )
                assistant_message_sql.code(sql, language="sql", line_numbers=True)
        else:
            assistant_message = st.chat_message(
                "assistant", avatar=avatar_url
            )
            assistant_message.write(sql)
            st.session_state["my_question"] = None  # Reset question
            st.stop()

        df = run_sql_cached(sql=sql)

        if df is not None:
            st.session_state["df"] = df

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                assistant_message_table = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                if len(df) > 10:
                    assistant_message_table.text("First 10 rows of data")
                    assistant_message_table.dataframe(df.head(10))
                else:
                    assistant_message_table.dataframe(df)

            if should_generate_chart_cached(question=my_question, sql=sql, df=df):

                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    assistant_message_plotly_code = st.chat_message(
                        "assistant",
                        avatar=avatar_url,
                    )
                    assistant_message_plotly_code.code(
                        code, language="python", line_numbers=True
                    )

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        assistant_message_chart = st.chat_message(
                            "assistant",
                            avatar=avatar_url,
                        )
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            assistant_message_chart.plotly_chart(fig)
                        else:
                            assistant_message_chart.error("I couldn't generate a chart")

            if st.session_state.get("show_summary", True):
                assistant_message_summary = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    assistant_message_summary.text(summary)

            if st.session_state.get("show_followup", True):
                assistant_message_followup = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                followup_questions = generate_followup_cached(
                    question=my_question, sql=sql, df=df
                )
                
                if len(followup_questions) > 0:
                    assistant_message_followup.text(
                        "Here are some possible follow-up questions"
                    )
                    # Print the first 5 follow-up questions
                    for question in followup_questions[:5]:
                        assistant_message_followup.button(question, on_click=set_question, args=(question,))
            
            # Clear question and df after displaying everything
            st.session_state["my_question"] = None
            st.session_state["df"] = None

    else:
        assistant_message_error = st.chat_message(
            "assistant", avatar=avatar_url
        )
        assistant_message_error.error("I wasn't able to generate SQL for that question")
        st.session_state["my_question"] = None  # Reset question