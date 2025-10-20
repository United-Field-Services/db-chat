import streamlit as st
from vanna_calls import setup_vanna
vn = setup_vanna()

# Test query
result = vn.run_sql("SELECT version();")
st.write("PostgreSQL version:", result)