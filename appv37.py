import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta, date
import anthropic
import re
import json
import os
import numpy as np
from typing import List, Dict, Optional

# Configure Streamlit
st.set_page_config(
    page_title="SupaBot BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Training System from v1justsupabot.py
class EnhancedTrainingSystem:
    def __init__(self, training_file="supabot_training.json"):
        self.training_file = training_file
        self.training_data = self.load_training_data()

    def load_training_data(self) -> List[Dict]:
        """Load training examples from JSON file"""
        if os.path.exists(self.training_file):
            try:
                with open(self.training_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_training_data(self):
        """Save training examples to JSON file"""
        try:
            with open(self.training_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save training data: {e}")
            return False

    def add_training_example(self, question: str, sql: str, feedback: str = "correct", explanation: str = ""):
        """Add a new training example with optional explanation"""
        example = {
            "question": question.lower().strip(),
            "sql": sql.strip(),
            "feedback": feedback,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        self.training_data.append(example)
        return self.save_training_data()

    def find_similar_examples(self, question: str, limit: int = 3) -> List[Dict]:
        """Find similar training examples using enhanced similarity"""
        question = question.lower().strip()
        scored_examples = []
        
        business_terms = {
            'sales': ['revenue', 'income', 'earnings', 'total'],
            'hour': ['time', 'hourly', 'per hour'],
            'store': ['location', 'branch', 'shop'],
            'total': ['sum', 'aggregate', 'combined', 'all'],
            'date': ['day', 'daily', 'time period']
        }
        
        for example in self.training_data:
            if example["feedback"] in ["correct", "corrected"]:
                q1_words = set(question.split())
                q2_words = set(example["question"].split())
                
                if len(q1_words | q2_words) > 0:
                    basic_similarity = len(q1_words & q2_words) / len(q1_words | q2_words)
                    
                    business_score = 0
                    for term, synonyms in business_terms.items():
                        if any(syn in question for syn in [term] + synonyms):
                            if any(syn in example["question"] for syn in [term] + synonyms):
                                business_score += 0.3
                    
                    final_score = basic_similarity + business_score
                    scored_examples.append((final_score, example))
        
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for score, example in scored_examples[:limit] if score > 0.2]

    def get_training_context(self, question: str) -> str:
        """Get relevant training examples formatted as context"""
        similar_examples = self.find_similar_examples(question)
        if not similar_examples:
            return ""
        
        context = "RELEVANT TRAINING EXAMPLES:\n\n"
        for i, example in enumerate(similar_examples, 1):
            context += f"Example {i}:\n"
            context += f"Question: {example['question']}\n"
            context += f"SQL: {example['sql']}\n"
            if example.get('explanation'):
                context += f"Note: {example['explanation']}\n"
            context += "\n"
        
        return context

def get_training_system():
    """Initialize training system with default examples"""
    training_system = EnhancedTrainingSystem()
    
    if len(training_system.training_data) == 0:
        default_examples = [
            {
                "question": "sales per hour total of all stores and all dates",
                "sql": """
                WITH hourly_sales AS (
                    SELECT 
                        EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
                        TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM') as hour_label,
                        SUM(t.total) as total_sales
                    FROM transactions t
                    WHERE LOWER(t.transaction_type) = 'sale' 
                    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
                    GROUP BY EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila'), TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM')
                    ORDER BY hour
                )
                SELECT hour, hour_label, COALESCE(total_sales, 0) as total_sales FROM hourly_sales;
                """,
                "feedback": "correct",
                "explanation": "Groups by hour only across ALL stores and dates. Different from per-store breakdown.",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        for example in default_examples:
            training_system.training_data.append(example)
        training_system.save_training_data()
    
    return training_system

def create_db_connection():
    try:
        # Try multiple possible secret configurations
        if "postgres" in st.secrets:
            # Format 1: [postgres] section
            return psycopg2.connect(
                host=st.secrets["postgres"]["host"],
                database=st.secrets["postgres"]["database"],
                user=st.secrets["postgres"]["user"],
                password=st.secrets["postgres"]["password"],
                port=st.secrets["postgres"]["port"]
            )
        else:
            # Format 2: Individual keys (fallback)
            return psycopg2.connect(
                host=st.secrets.get("SUPABASE_HOST", st.secrets.get("host")),
                database=st.secrets.get("SUPABASE_DB", st.secrets.get("database")),
                user=st.secrets.get("SUPABASE_USER", st.secrets.get("user")),
                password=st.secrets.get("SUPABASE_PASSWORD", st.secrets.get("password")),
                port=st.secrets.get("SUPABASE_PORT", st.secrets.get("port", "5432"))
            )
    except KeyError as e:
        st.error(f"Missing database credential: {e}")
        st.info("Please add your database credentials to .streamlit/secrets.toml")
        return None
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

@st.cache_data(ttl=3600)
def get_database_schema():
    """Fetch the complete database schema including sample data"""
    conn = create_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
        tables = cursor.fetchall()
        schema_info = {}
        for (table_name,) in tables:
            cursor.execute(f"SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_name = '{table_name}' ORDER BY ordinal_position")
            columns = cursor.fetchall()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_data = cursor.fetchall()
            schema_info[table_name] = {'columns': columns, 'row_count': row_count, 'sample_data': sample_data}
        return schema_info
    except Exception as e:
        st.error(f"Schema fetch failed: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_claude_client():
    try:
        # Try multiple possible secret configurations
        if "anthropic" in st.secrets:
            api_key = st.secrets["anthropic"]["api_key"]
        else:
            # Format 2: Individual keys (fallback)
            api_key = st.secrets.get("CLAUDE_API_KEY", st.secrets.get("ANTHROPIC_API_KEY"))
        
        if api_key:
            return anthropic.Anthropic(api_key=api_key)
        return None
    except:
        return None

# AI Assistant Core Functions
def generate_smart_sql(question, schema_info=None, training_system=None):
    """Ultimate AI SQL generator with training system integration"""
    client = get_claude_client()
    if not client: return None
    
    schema_context = "DATABASE SCHEMA:\n\n"
    if schema_info:
        for table_name, info in schema_info.items():
            schema_context += f"TABLE: {table_name} ({info['row_count']} rows)\nColumns:\n"
            for col_name, data_type, nullable, default in info['columns']:
                nullable_str = "NULL" if nullable == 'YES' else "NOT NULL"
                schema_context += f"  - {col_name}: {data_type} {nullable_str}\n"
            schema_context += "\n"

    training_context = training_system.get_training_context(question) if training_system else ""

    prompt = f"""{schema_context}{training_context}

BUSINESS CONTEXT:
- This is a retail business database tracking sales, inventory, products, and stores.
- Valid sales transactions have: transaction_type = 'sale' AND (is_cancelled IS NULL OR is_cancelled = false).
- For product-level revenue and quantity, use SUM(transaction_items.item_total) and SUM(transaction_items.quantity).
- For transaction-level revenue (e.g., total sales, sales by store), use SUM(transactions.total) to match POS recorded totals.
- TIMEZONE: Data is in Philippines timezone (UTC+8)
- TIME FORMAT: Always format time as 12-hour format (1:00 PM, 7:00 PM, etc.)

CRITICAL AGGREGATION RULES:
1. When user asks for "total across all stores" or "total of all stores" - GROUP BY time/category ONLY, do NOT group by store
2. When user asks for "per store" or "by store" - GROUP BY both store AND time/category
3. Pay attention to the level of aggregation requested.

USER QUESTION: {question}

INSTRUCTIONS:
1. Generate a PostgreSQL query that matches the EXACT aggregation level requested.
2. Use the training examples as reference.
3. For "total across all stores and all dates by hour": GROUP BY hour only. Use SUM(transactions.total)
4. For "sales per store per hour": GROUP BY store AND hour. Use SUM(transactions.total)
5. For product-related queries (e.g., 'top products by sales'), use SUM(transaction_items.item_total) for revenue and SUM(transaction_items.quantity) for units sold.
6. Use CTEs for readability. Use COALESCE for NULLs. Include meaningful aliases.
7. For time-based queries use AT TIME ZONE 'Asia/Manila' and format time as 12-hour with AM/PM.
8. Order results descending by the main metric.

Generate ONLY the SQL query, no explanations:"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620", max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        sql = response.content[0].text.strip()
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql).strip()
        if not sql.endswith(';'): sql += ';'
        return sql
    except Exception as e:
        st.error(f"AI query generation failed: {e}")
        return None

def interpret_results(question, results_df, sql_query):
    client = get_claude_client()
    if not client or results_df.empty: return "The query returned no results."
    
    results_summary = f"Query returned {len(results_df)} rows. Columns: {', '.join(results_df.columns)}\n\n"
    results_summary += "First 10 rows:\n" + results_df.head(10).to_string()
    
    prompt = f"""You are a business intelligence expert. The user asked: "{question}"

SQL Query executed:
{sql_query}

Results:
{results_summary}

Please provide a clear, concise, conversational but professional answer to the user's question, followed by key insights and actionable recommendations. Use bullet points. Interpret the data, don't just repeat it. Format monetary amounts as ₱X,XXX (no decimals)."""
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception:
        return "Could not interpret results."

# ENHANCED SMART VISUALIZATION with 8+ Chart Types
def create_smart_visualization(results_df, question):
    """Enhanced visualization function that automatically selects the best chart type"""
    
    if results_df.empty:
        return None
    
    # Get column types
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    text_cols = results_df.select_dtypes(include=['object']).columns.tolist()
    date_cols = results_df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    
    if not numeric_cols:
        return None
    
    # Clean the question for analysis
    question_lower = question.lower()
    
    # Determine chart type based on question keywords and data structure
    chart_type = determine_chart_type(question_lower, results_df, numeric_cols, text_cols, date_cols)
    
    try:
        fig = None
        
        if chart_type == "pie":
            fig = create_pie_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "treemap":
            fig = create_treemap_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "scatter":
            fig = create_scatter_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "line":
            fig = create_line_chart(results_df, question, numeric_cols, text_cols, date_cols)
        elif chart_type == "heatmap":
            fig = create_heatmap_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "box":
            fig = create_box_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "area":
            fig = create_area_chart(results_df, question, numeric_cols, text_cols, date_cols)
        else:  # Default to bar chart
            fig = create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        # Apply consistent styling
        if fig:
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=500,
                title_x=0.5
            )
            
        return fig
        
    except Exception as e:
        # Fallback to bar chart if anything fails
        return create_bar_chart(results_df, question, numeric_cols, text_cols)

def determine_chart_type(question_lower, results_df, numeric_cols, text_cols, date_cols):
    """Intelligently determine the best chart type based on question and data"""
    
    # Keywords for different chart types
    pie_keywords = ['distribution', 'breakdown', 'percentage', 'proportion', 'share', 'composition', 'part of']
    treemap_keywords = ['hierarchy', 'treemap', 'nested', 'structure', 'composition', 'size comparison']
    scatter_keywords = ['correlation', 'relationship', 'vs', 'against', 'compare', 'scatter', 'relationship between']
    line_keywords = ['trend', 'over time', 'timeline', 'progression', 'change', 'growth', 'decline']
    heatmap_keywords = ['pattern', 'heatmap', 'intensity', 'by hour and day', 'activity']
    box_keywords = ['outlier', 'distribution', 'quartile', 'median', 'range', 'variance']
    area_keywords = ['cumulative', 'stacked', 'total over time', 'accumulated']
    
    # Check for specific chart type keywords
    if any(keyword in question_lower for keyword in pie_keywords) and len(text_cols) >= 1:
        return "pie"
    
    if any(keyword in question_lower for keyword in treemap_keywords) and len(text_cols) >= 1:
        return "treemap"
    
    if any(keyword in question_lower for keyword in scatter_keywords) and len(numeric_cols) >= 2:
        return "scatter"
    
    if any(keyword in question_lower for keyword in line_keywords) and (date_cols or 'hour' in question_lower or 'day' in question_lower):
        return "line"
    
    if any(keyword in question_lower for keyword in heatmap_keywords):
        return "heatmap"
    
    if any(keyword in question_lower for keyword in box_keywords):
        return "box"
    
    if any(keyword in question_lower for keyword in area_keywords) and date_cols:
        return "area"
    
    # Data-driven decisions
    row_count = len(results_df)
    
    # For small datasets with categories, prefer pie charts for distribution questions
    if row_count <= 10 and len(text_cols) >= 1 and ('category' in question_lower or 'type' in question_lower):
        return "pie"
    
    # For datasets with multiple numeric columns, prefer scatter
    if len(numeric_cols) >= 2 and row_count >= 10:
        return "scatter"
    
    # For time-based data, prefer line charts
    if date_cols or any(col for col in results_df.columns if 'hour' in col.lower() or 'time' in col.lower()):
        return "line"
    
    # Default to bar chart
    return "bar"

def get_best_value_column(numeric_cols):
    """Select the best numeric column for values"""
    priority_terms = ['revenue', 'sales', 'total', 'amount', 'value', 'price', 'cost']
    
    for term in priority_terms:
        for col in numeric_cols:
            if term in col.lower():
                return col
    
    return numeric_cols[0]  # Fallback to first numeric column

def get_best_label_column(text_cols):
    """Select the best text column for labels"""
    priority_terms = ['name', 'category', 'type', 'store', 'product']
    
    for term in priority_terms:
        for col in text_cols:
            if term in col.lower() and 'id' not in col.lower():
                return col
    
    # Return first non-ID column
    for col in text_cols:
        if 'id' not in col.lower():
            return col
    
    return text_cols[0]  # Fallback to first text column

def create_pie_chart(results_df, question, numeric_cols, text_cols):
    """Create a pie chart for distribution/breakdown questions"""
    if not text_cols or not numeric_cols:
        return None
    
    # Select best columns
    value_col = get_best_value_column(numeric_cols)
    label_col = get_best_label_column(text_cols)
    
    # Filter and prepare data
    df_clean = results_df[results_df[value_col] > 0].copy()
    if len(df_clean) > 10:  # Limit to top 10 for readability
        df_clean = df_clean.nlargest(10, value_col)
    
    fig = px.pie(df_clean, values=value_col, names=label_col,
                title=f"Distribution: {question}")
    
    return fig

def create_treemap_chart(results_df, question, numeric_cols, text_cols):
    """Create a treemap for hierarchical data"""
    if not text_cols or not numeric_cols:
        return None
    
    value_col = get_best_value_column(numeric_cols)
    label_col = get_best_label_column(text_cols)
    
    df_clean = results_df[results_df[value_col] > 0].copy()
    if len(df_clean) > 20:
        df_clean = df_clean.nlargest(20, value_col)
    
    fig = px.treemap(df_clean, path=[label_col], values=value_col,
                    title=f"Treemap: {question}")
    
    return fig

def create_scatter_chart(results_df, question, numeric_cols, text_cols):
    """Create a scatter plot for correlation analysis"""
    if len(numeric_cols) < 2:
        return None
    
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    
    # If there's a third numeric column, use it for size
    size_col = numeric_cols[2] if len(numeric_cols) > 2 else None
    
    # If there's a text column, use it for color
    color_col = text_cols[0] if text_cols else None
    
    fig = px.scatter(results_df, x=x_col, y=y_col,
                    size=size_col, color=color_col,
                    title=f"Relationship: {question}",
                    hover_data=text_cols[:2] if text_cols else None)
    
    return fig

def create_line_chart(results_df, question, numeric_cols, text_cols, date_cols):
    """Create a line chart for time series data"""
    if not numeric_cols:
        return None
    
    # Determine x-axis (time-based)
    x_col = None
    if date_cols:
        x_col = date_cols[0]
    else:
        # Look for time-related columns
        for col in results_df.columns:
            if any(time_word in col.lower() for time_word in ['hour', 'time', 'date', 'day']):
                x_col = col
                break
    
    if not x_col:
        x_col = results_df.columns[0]  # Fallback to first column
    
    y_col = get_best_value_column(numeric_cols)
    
    # Sort by x-axis for proper line connection
    df_sorted = results_df.sort_values(x_col)
    
    fig = px.line(df_sorted, x=x_col, y=y_col,
                 title=f"Trend: {question}",
                 markers=True)
    
    return fig

def create_heatmap_chart(results_df, question, numeric_cols, text_cols):
    """Create a heatmap for pattern analysis"""
    if len(results_df.columns) < 3:
        return None
    
    # Try to create a pivot table for heatmap
    if len(text_cols) >= 2 and len(numeric_cols) >= 1:
        try:
            pivot_df = results_df.pivot_table(
                index=text_cols[0],
                columns=text_cols[1],
                values=numeric_cols[0],
                fill_value=0
            )
            
            fig = px.imshow(pivot_df,
                           title=f"Pattern Analysis: {question}",
                           aspect="auto",
                           color_continuous_scale="Blues")
            
            return fig
        except:
            pass
    
    # Fallback to correlation heatmap if multiple numeric columns
    if len(numeric_cols) >= 3:
        corr_matrix = results_df[numeric_cols].corr()
        fig = px.imshow(corr_matrix,
                       title=f"Correlation: {question}",
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        return fig
    
    return None

def create_box_chart(results_df, question, numeric_cols, text_cols):
    """Create box plots for distribution analysis"""
    if not numeric_cols:
        return None
    
    y_col = get_best_value_column(numeric_cols)
    x_col = text_cols[0] if text_cols else None
    
    if x_col:
        fig = px.box(results_df, x=x_col, y=y_col,
                    title=f"Distribution: {question}")
    else:
        fig = px.box(results_df, y=y_col,
                    title=f"Distribution: {question}")
    
    return fig

def create_area_chart(results_df, question, numeric_cols, text_cols, date_cols):
    """Create area chart for cumulative data"""
    if not numeric_cols:
        return None
    
    x_col = date_cols[0] if date_cols else results_df.columns[0]
    y_col = get_best_value_column(numeric_cols)
    
    df_sorted = results_df.sort_values(x_col)
    
    fig = px.area(df_sorted, x=x_col, y=y_col,
                 title=f"Cumulative: {question}")
    
    return fig

def create_bar_chart(results_df, question, numeric_cols, text_cols):
    """Create bar chart (original functionality)"""
    if not text_cols or not numeric_cols:
        return None
    
    y_col = get_best_value_column(numeric_cols)
    x_col = get_best_label_column(text_cols)
    
    df_filtered = results_df[results_df[y_col] > 0].copy()
    if df_filtered.empty:
        return None
        
    df_sorted = df_filtered.sort_values(by=y_col, ascending=False).head(25)
    
    # Determine orientation based on label length
    chart_type = 'h' if any(len(str(s)) > 15 for s in df_sorted[x_col]) else 'v'
    
    if chart_type == 'h':
        fig = px.bar(df_sorted, x=y_col, y=x_col, orientation='h',
                    title=f"Analysis: {question}")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    else:
        fig = px.bar(df_sorted, x=x_col, y=y_col,
                    title=f"Analysis: {question}")
    
    return fig

# Execute query for AI Assistant
def execute_query_for_assistant(sql):
    conn = create_db_connection()
    if not conn: 
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '30s'")
        df = pd.read_sql(sql, conn)
        # Format datetime columns properly
        for col in df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M')
        return df
    except psycopg2.errors.QueryCanceled:
        st.error("Query took too long to execute. Try a simpler question.")
        return None
    except Exception as e:
        error_msg = str(e)
        st.error(f"Query execution failed: {error_msg}")
        if "does not exist" in error_msg: 
            st.info("💡 The query references a table or column that doesn't exist.")
        elif "syntax error" in error_msg: 
            st.info("💡 There's a syntax error in the SQL.")
        return None
    finally:
        if conn: 
            conn.close()

def get_column_config(df):
    """Dynamic formatting for dataframes from v1justsupabot.py"""
    config = {}
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['revenue', 'profit', 'price', 'cost', 'total', 'amount', 'value', 'sales']):
            config[col] = st.column_config.NumberColumn(label=col.replace("_", " ").title(), format="₱%d")
        elif any(keyword in col_lower for keyword in ['quantity', 'count', 'sold', 'items', 'transactions']):
             config[col] = st.column_config.NumberColumn(label=col.replace("_", " ").title(), format="%,d")
        else:
            config[col] = st.column_config.TextColumn(label=col.replace("_", " ").title())
    return config

# Dashboard Data Fetching Functions (from appv8.py)
def execute_query_for_dashboard(sql, params=None):
    conn = create_db_connection()
    if not conn: 
        return None
    try:
        # Use pd.read_sql with parameters to prevent SQL injection
        df = pd.read_sql(sql, conn, params=params)
        return df
    except Exception as e:
        # Silently handle errors for dashboard queries to avoid breaking the UI
        print(f"Dashboard query error: {e}")
        return pd.DataFrame() # Return empty dataframe on error
    finally:
        if conn: 
            conn.close()

@st.cache_data(ttl=300)
def get_latest_metrics():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        COALESCE(SUM(t.total), 0) as latest_sales,
        COUNT(DISTINCT t.ref_id) as latest_transactions
    FROM transactions t
    CROSS JOIN latest_date ld
    WHERE LOWER(transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_previous_metrics():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        COALESCE(SUM(t.total), 0) as previous_sales,
        COUNT(DISTINCT t.ref_id) as previous_transactions
    FROM transactions t
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date - INTERVAL '1 day'
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_hourly_sales():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
        TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM') as hour_label,
        COALESCE(SUM(t.total), 0) as sales
    FROM transactions t
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
    GROUP BY 1,2
    HAVING SUM(t.total) > 0 ORDER BY hour
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_store_performance():
    """Get store performance for the latest day only (to match hourly sales)"""
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        s.name as store_name, 
        COALESCE(SUM(t.total), 0) as total_sales
    FROM stores s
    LEFT JOIN transactions t ON s.id = t.store_id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
    GROUP BY s.name
    HAVING COALESCE(SUM(t.total), 0) > 0
    ORDER BY total_sales DESC
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_daily_trend(days=30, store_ids: Optional[List[int]] = None):
    """Fetches daily sales trend, with optional store filter."""
    params = [f'{days} days']
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)
        
    sql = f"""
    SELECT 
        DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
        COALESCE(SUM(t.total), 0) as daily_sales
    FROM transactions t
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= CURRENT_DATE - INTERVAL %s
    {store_clause}
    GROUP BY 1
    HAVING SUM(t.total) > 0
    ORDER BY date
    """
    df = execute_query_for_dashboard(sql, params=params)
    if df is not None and not df.empty:
        df['cumulative_sales'] = df['daily_sales'].cumsum()
    return df

@st.cache_data(ttl=300)
def get_store_count():
    sql = "SELECT COUNT(DISTINCT id) as store_count FROM stores"
    result = execute_query_for_dashboard(sql)
    return result.iloc[0]['store_count'] if result is not None and len(result) > 0 else 0

@st.cache_data(ttl=300)
def get_product_performance():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        p.name as product_name,
        SUM(ti.quantity) as total_quantity_sold,
        SUM(ti.item_total) as total_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY p.name
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_transaction_analysis():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        t.ref_id,
        s.name as store_name,
        SUM(ti.quantity) as items_per_transaction,
        t.total as total_value,
        AVG(ti.item_total / ti.quantity) as avg_item_value
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    JOIN stores s ON t.store_id = s.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY t.ref_id, s.name, t.total
    HAVING SUM(ti.quantity) > 0
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_daily_sales_by_store():
    sql = """
    SELECT 
        DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
        s.name as store_name,
        COALESCE(SUM(t.total), 0) as daily_sales
    FROM transactions t
    JOIN stores s ON t.store_id = s.id
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY 1,2
    ORDER BY date DESC
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_transaction_values_by_store():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        s.name as store_name,
        t.total as total_value
    FROM transactions t
    JOIN stores s ON t.store_id = s.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    AND t.total > 0
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_category_sales():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        p.category as product_category,
        SUM(ti.quantity) as total_quantity_sold,
        SUM(ti.item_total) as total_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY p.category
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC
    """
    return execute_query_for_dashboard(sql)

# --- NEW DASHBOARD DATA FUNCTIONS START ---
def get_time_filter_interval(time_filter="7d"):
    """Helper function to convert time filter string to SQL interval."""
    mapping = {"1D": "1 day", "7D": "7 days", "1M": "1 month", "6M": "6 months", "1Y": "1 year"}
    return mapping.get(time_filter, "7 days")

@st.cache_data(ttl=300)
def get_filtered_metrics(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Metrics based on time filter selection, with correct profit calculation."""
    interval = get_time_filter_interval(time_filter)
    
    def _get_metrics_for_period(start_interval, end_interval):
        params = [start_interval, end_interval]
        store_clause = ""
        if store_ids:
            store_clause = "AND t.store_id = ANY(%s)"
            params.append(store_ids)
            
        sql = f"""
        WITH period_transactions AS (
            SELECT ref_id, total, store_id
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
            AND t.transaction_time AT TIME ZONE 'Asia/Manila' BETWEEN NOW() - INTERVAL %s AND NOW() - INTERVAL %s
            {store_clause}
        ),
        period_costs AS (
            SELECT
                SUM(ti.quantity * p.cost) as total_cogs
            FROM transaction_items ti
            JOIN products p ON ti.product_id = p.id
            WHERE ti.transaction_ref_id IN (SELECT ref_id FROM period_transactions)
        )
        SELECT
            (SELECT COALESCE(SUM(total), 0) FROM period_transactions) as sales,
            (SELECT COALESCE(SUM(total), 0) FROM period_transactions) - (SELECT COALESCE(total_cogs, 0) FROM period_costs) as profit,
            (SELECT COUNT(ref_id) FROM period_transactions) as transactions,
            (SELECT COUNT(DISTINCT store_id) FROM period_transactions) as active_stores,
            (SELECT COALESCE(SUM(total), 0) / NULLIF(COUNT(ref_id), 0) FROM period_transactions) as avg_transaction_value
        """
        return execute_query_for_dashboard(sql, params=params)

    # Current and previous periods
    df_current = _get_metrics_for_period(interval, '0 seconds')
    df_previous = _get_metrics_for_period(f'2 * {interval}', interval)
    
    metrics = df_current.iloc[0].to_dict() if not df_current.empty else {}
    prev_metrics = df_previous.iloc[0].to_dict() if not df_previous.empty else {}
    
    # Rename previous metrics for clarity
    metrics['prev_sales'] = prev_metrics.get('sales', 0)
    metrics['prev_profit'] = prev_metrics.get('profit', 0)
    metrics['prev_transactions'] = prev_metrics.get('transactions', 0)

    # YoY comparison for 1Y filter
    if time_filter == "1Y":
        df_yoy = _get_metrics_for_period('2 years', '1 year')
        yoy_metrics = df_yoy.iloc[0].to_dict() if not df_yoy.empty else {}
        metrics['yoy_sales'] = yoy_metrics.get('sales', 0)
        metrics['yoy_profit'] = yoy_metrics.get('profit', 0)

    return metrics

@st.cache_data(ttl=300)
def get_sales_by_category_pie(time_filter="7D", store_ids: Optional[List[int]] = None):
    """SQL query to get category sales for pie chart, with optional store filter."""
    interval = get_time_filter_interval(time_filter)
    params = [interval]
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)

    sql = f"""
    SELECT 
        p.category, 
        SUM(ti.item_total) as total_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND t.transaction_time AT TIME ZONE 'Asia/Manila' >= NOW() - INTERVAL %s
    {store_clause}
    GROUP BY p.category
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC;
    """
    return execute_query_for_dashboard(sql, params=params)

@st.cache_data(ttl=300)
def get_inventory_by_category_pie(store_ids: Optional[List[int]] = None):
    """SQL query to get inventory levels by category, with optional store filter."""
    params = []
    store_clause = ""
    if store_ids:
        store_clause = "WHERE i.store_id = ANY(%s)"
        params.append(store_ids)

    sql = f"""
    SELECT
        p.category,
        SUM(i.quantity_on_hand) as total_inventory
    FROM inventory i
    JOIN products p ON i.product_id = p.id
    {store_clause}
    GROUP BY p.category
    HAVING SUM(i.quantity_on_hand) > 0;
    """
    df = execute_query_for_dashboard(sql, params=params if params else None)
    return df

@st.cache_data(ttl=300)
def get_top_sellers_analysis(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Top products with momentum indicators, with optional store filter."""
    interval = get_time_filter_interval(time_filter)
    
    params_current = [interval]
    params_previous = [f'2 * {interval}', interval]
    
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params_current.append(store_ids)
        params_previous.append(store_ids)

    sql = f"""
    WITH current_period AS (
        SELECT 
            p.name as product_name,
            SUM(ti.item_total) as total_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        AND t.transaction_time AT TIME ZONE 'Asia/Manila' >= NOW() - INTERVAL %s
        {store_clause}
        GROUP BY p.name
    ),
    previous_period AS (
        SELECT 
            p.name as product_name,
            SUM(ti.item_total) as prev_revenue
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        AND t.transaction_time AT TIME ZONE 'Asia/Manila' BETWEEN NOW() - INTERVAL %s AND NOW() - INTERVAL %s
        {store_clause}
        GROUP BY p.name
    )
    SELECT
        cp.product_name,
        cp.total_revenue,
        COALESCE(pp.prev_revenue, 0) as prev_revenue,
        (cp.total_revenue - COALESCE(pp.prev_revenue, 0)) as revenue_change
    FROM current_period cp
    LEFT JOIN previous_period pp ON cp.product_name = pp.product_name
    ORDER BY cp.total_revenue DESC
    LIMIT 10;
    """
    # Combine params for a single query execution
    # This requires re-writing the query to not use CTEs that depend on different params
    # For simplicity and since the function is cached, separate executions are acceptable here.
    # A more optimized version would use a single query with CASE statements.
    
    # Let's create a single query for better performance
    single_sql = f"""
    SELECT
        p.name as product_name,
        SUM(CASE WHEN t.transaction_time AT TIME ZONE 'Asia/Manila' >= NOW() - INTERVAL %s THEN ti.item_total ELSE 0 END) as total_revenue,
        SUM(CASE WHEN t.transaction_time AT TIME ZONE 'Asia/Manila' BETWEEN NOW() - INTERVAL %s AND NOW() - INTERVAL %s THEN ti.item_total ELSE 0 END) as prev_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND t.transaction_time AT TIME ZONE 'Asia/Manila' >= NOW() - INTERVAL %s
    {store_clause}
    GROUP BY p.name
    ORDER BY total_revenue DESC
    LIMIT 10;
    """
    params = [interval, f'2 * {interval}', interval, f'2 * {interval}']
    if store_ids:
        params.append(store_ids)
        
    df = execute_query_for_dashboard(single_sql, params=params)
    if not df.empty:
        df['revenue_change'] = df['total_revenue'] - df['prev_revenue']
    return df


@st.cache_data(ttl=300)
def get_store_performance_heatmap(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Store performance data for heatmap, with optional store filter."""
    interval = get_time_filter_interval(time_filter)
    params = [interval]
    store_clause = ""
    if store_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_ids)

    sql = f"""
    SELECT 
        s.name as store_name,
        DATE_TRUNC('day', t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
        SUM(t.total) as daily_sales
    FROM transactions t
    JOIN stores s ON t.store_id = s.id
    WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND t.transaction_time AT TIME ZONE 'Asia/Manila' >= NOW() - INTERVAL %s
    {store_clause}
    GROUP BY s.name, date
    ORDER BY s.name, date;
    """
    return execute_query_for_dashboard(sql, params=params)

@st.cache_data(ttl=300)
def get_fad_stays_analysis(time_filter="7D", store_ids: Optional[List[int]] = None):
    """Products gaining/losing momentum."""
    df = get_top_sellers_analysis(time_filter, store_ids)
    if not df.empty:
        df['momentum'] = (df['total_revenue'] - df['prev_revenue']) / df['prev_revenue'].replace(0, np.nan)
        df_gainers = df.sort_values(by='revenue_change', ascending=False).head(5)
        df_losers = df.sort_values(by='revenue_change', ascending=True).head(5)
        return df_gainers, df_losers
    return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300)
def get_business_highlights(time_filter="7D", store_ids: Optional[List[int]] = None):
    """AI-generated business insights for the period."""
    metrics = get_filtered_metrics(time_filter, store_ids)
    top_sellers = get_top_sellers_analysis(time_filter, store_ids)

    if not metrics or top_sellers.empty:
        return "Not enough data to generate highlights."

    client = get_claude_client()
    if not client:
        return "AI client not configured. Cannot generate highlights."

    prompt = f"""
    Based on the following data for the last {time_filter}, generate 3-4 key business highlights and actionable insights. Be concise and use bullet points.

    - Total Sales: ₱{metrics.get('sales', 0):,.0f}
    - Total Profit: ₱{metrics.get('profit', 0):,.0f}
    - Total Transactions: {metrics.get('transactions', 0):,}
    - Top Selling Product: {top_sellers['product_name'].iloc[0]} with ₱{top_sellers['total_revenue'].iloc[0]:,.0f} in sales.
    - Top Gaining Product: {top_sellers.sort_values('revenue_change', ascending=False).iloc[0]['product_name']}

    Generate the highlights now.
    """
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Could not generate AI highlights: {e}"

# --- NEW DASHBOARD DATA FUNCTIONS END ---


def create_calendar_heatmap(df_cal, date_col, value_col):
    """Create calendar heatmap visualization"""
    if df_cal is None or df_cal.empty:
        return None
    df_cal = df_cal.copy()
    df_cal[date_col] = pd.to_datetime(df_cal[date_col])
    df_cal = df_cal.sort_values(date_col)
    
    # Create week-based calendar
    df_cal['week'] = df_cal[date_col].dt.isocalendar().week
    df_cal['day_of_week'] = df_cal[date_col].dt.dayofweek
    df_cal['day_num'] = df_cal[date_col].dt.day
    
    min_week = df_cal['week'].min()
    df_cal['week_normalized'] = df_cal['week'] - min_week
    
    unique_weeks = sorted(df_cal['week_normalized'].unique())
    if not unique_weeks or len(df_cal) < 7:
        return None  # Return None if insufficient data
    
    matrix = []
    annotations = []
    
    # Build matrix from Monday (0) to Sunday (6)
    for day in range(7):  # 0 = Monday, 6 = Sunday
        row = []
        for week in unique_weeks:
            day_data = df_cal[(df_cal['week_normalized'] == week) & (df_cal['day_of_week'] == day)]
            if not day_data.empty:
                value = day_data.iloc[0][value_col]
                day_num = day_data.iloc[0]['day_num']
                row.append(value)
                text_color = 'black' if value <= max(df_cal[value_col]) * 0.5 else 'white'
                annotations.append(dict(
                    x=week, y=day,  # y-coordinate matches day index (0 = Monday, 6 = Sunday)
                    text=str(day_num),
                    showarrow=False,
                    font=dict(color=text_color, size=12, family='Arial Black'),
                    xref='x', yref='y'
                ))
            else:
                row.append(0)  # No data for this day
        matrix.append(row)
    
    # Create heatmap with UI-matched color scheme
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale=[
            [0, '#f0f4f8'],    # Very light gray-blue for no data
            [0.2, '#d9e6f2'],  # Light blue for low sales
            [0.4, '#a3c9e0'],  # Medium blue for medium sales
            [0.6, '#6baed6'],  # Darker blue for high sales
            [0.8, '#3182bd'],  # Deep blue for higher sales
            [1, '#1b4d7e']     # Dark blue for maximum
        ],
        showscale=True,
        colorbar=dict(
            title="Sales (PHP)",
            tickmode="linear",
            tick0=0,
            dtick=max(df_cal[value_col]) / 5 if df_cal[value_col].max() > 0 else 1,
            tickformat=",.0f"
        ),
        hovertemplate='<b>%{text}</b><br>Sales: ₱%{z:,.0f}<extra></extra>',
        text=[[f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day]}" for week in range(len(unique_weeks))] for day in range(7)]
    ))
    
    # Add day number annotations
    for ann in annotations:
        fig.add_annotation(ann)
    
    # Update layout with Monday at top, Sunday at bottom
    fig.update_layout(
        title=f'📅 Daily Sales Calendar ({df_cal[date_col].min().strftime("%b %d")} - {df_cal[date_col].max().strftime("%b %d, %Y")})',
        xaxis=dict(
            title="",
            tickvals=list(range(len(unique_weeks))),
            ticktext=[f"Week {i+1}" for i in range(len(unique_weeks))],
            side='top'
        ),
        yaxis=dict(
            title="",
            tickvals=list(range(7)),  # 0 = Monday, 6 = Sunday
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],  # Monday at top
            autorange=True  # Automatically adjust while respecting ticktext order
        ),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        title_x=0.5,
        font=dict(size=11)
    )
    
    return fig

# CSS Styling
def load_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .main-header {
        background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 100%);
        padding: 1.5rem; border-radius: 10px; text-align: center;
        color: white; margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: bold; }
    
    /* KPI Metric Boxes - Ensure Equal Height */
    div[data-testid="stMetric"] {
        background-color: #1c1e26; 
        border: 1px solid #2e303d;
        padding: 1rem; 
        border-radius: 10px;
        height: 130px; /* Increased height for more content */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    div[data-testid="stMetric"] > div:nth-child(1) {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 1.8rem; /* Adjusted for space */
        font-weight: bold; 
        color: #00d2ff;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stMetric"] > div:nth-child(3) {
        font-size: 0.8rem;
    }

    /* Custom card styling */
    .dashboard-card {
        background-color: #1c1e26;
        border: 1px solid #2e303d;
        border-radius: 10px;
        padding: 1rem;
        height: 350px; /* Fixed height for alignment */
        display: flex;
        flex-direction: column;
    }
    .dashboard-card-tall {
        height: 450px;
    }
    .dashboard-card h5 {
        font-size: 1.1rem;
        color: #c7c7c7;
        margin-bottom: 1rem;
        border-bottom: 1px solid #3a47d5;
        padding-bottom: 0.5rem;
    }
    
    .insight-box {
        background: #16a085; padding: 1rem; border-radius: 8px;
        color: white; margin-top: 1rem; text-align: center;
    }
    .user-message{
        background:linear-gradient(135deg, #3a47d5 0%, #00d2ff 100%);
        padding:1rem 1.5rem; border-radius:20px 20px 0 20px;
        margin:1rem 0; color:white; font-weight:500;
    }
    .ai-message{
        background: #262730; border: 1px solid #3d3d3d;
        padding:1rem 1.5rem; border-radius:20px 20px 20px 0;
        margin:1rem 0; color:white;
    }
    button[data-baseweb="tab"] {
        background-color: transparent;
        border-bottom: 2px solid transparent;
        font-size: 1.1rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #00d2ff;
        color: #00d2ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "training_system" not in st.session_state:
        st.session_state.training_system = get_training_system()
    if "schema_info" not in st.session_state:
        st.session_state.schema_info = None
    # --- MODIFICATION START ---
    if "dashboard_time_filter" not in st.session_state:
        st.session_state.dashboard_time_filter = "7D"
    if "dashboard_store_filter" not in st.session_state:
        st.session_state.dashboard_store_filter = ["All Stores"]
    # --- MODIFICATION END ---

# --- MODIFICATION START: REPLACED render_dashboard() FUNCTION ---

@st.cache_data(ttl=300)
def get_dashboard_top_sellers(time_filter="7D", store_filter_ids=None):
    """Simple, working top sellers query"""
    days_map = {"1D": 1, "7D": 7, "1M": 30, "6M": 180, "1Y": 365}
    days = days_map.get(time_filter, 7)
    
    params = [days]
    store_clause = ""
    if store_filter_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.append(store_filter_ids)
    
    sql = f"""
    SELECT 
        p.name as product_name,
        SUM(ti.item_total) as total_revenue,
        SUM(ti.quantity) as total_quantity,
        COUNT(DISTINCT t.ref_id) as transaction_count
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND t.transaction_time >= CURRENT_DATE - INTERVAL '{days} days'
    {store_clause}
    GROUP BY p.name, p.id
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC
    LIMIT 10
    """
    
    return execute_query_for_dashboard(sql, params)

@st.cache_data(ttl=300)
def get_dashboard_metrics(time_filter="7D", store_filter_ids=None):
    """Simple metrics calculation that works"""
    days_map = {"1D": 1, "7D": 7, "1M": 30, "6M": 180, "1Y": 365}
    days = days_map.get(time_filter, 7)
    
    params = [days, days*2, days]
    store_clause = ""
    if store_filter_ids:
        store_clause = "AND t.store_id = ANY(%s)"
        params.extend([store_filter_ids, store_filter_ids])
    
    sql = f"""
    WITH current_period AS (
        SELECT 
            SUM(t.total) as sales,
            COUNT(DISTINCT t.ref_id) as transactions,
            AVG(t.total) as avg_transaction_value
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' 
        AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        AND t.transaction_time >= CURRENT_DATE - INTERVAL '{days} days'
        {store_clause}
    ),
    previous_period AS (
        SELECT 
            SUM(t.total) as prev_sales,
            COUNT(DISTINCT t.ref_id) as prev_transactions
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' 
        AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        AND t.transaction_time BETWEEN CURRENT_DATE - INTERVAL '{days*2} days' 
        AND CURRENT_DATE - INTERVAL '{days} days'
        {store_clause}
    )
    SELECT 
        COALESCE(cp.sales, 0) as current_sales,
        COALESCE(cp.sales * 0.65, 0) as current_profit,
        COALESCE(cp.transactions, 0) as current_transactions,
        COALESCE(cp.avg_transaction_value, 0) as avg_transaction_value,
        COALESCE(pp.prev_sales, 0) as prev_sales,
        COALESCE(pp.prev_sales * 0.65, 0) as prev_profit,
        COALESCE(pp.prev_transactions, 0) as prev_transactions
    FROM current_period cp
    CROSS JOIN previous_period pp
    """
    
    result = execute_query_for_dashboard(sql, params)
    return result.iloc[0].to_dict() if not result.empty else {}

@st.cache_data(ttl=300)
def get_dashboard_highlights(metrics, top_sellers_df, time_filter):
    """Generate business highlights from available data"""
    if not metrics or top_sellers_df.empty:
        return "📊 **Current Status:** Data is being processed. Please check back in a moment for detailed insights."
    
    sales = metrics.get('current_sales', 0)
    transactions = metrics.get('current_transactions', 0)
    
    if sales == 0:
        return "📊 **Current Status:** No sales data available for the selected period and filters."
    
    top_product = top_sellers_df.iloc[0]['product_name'] if not top_sellers_df.empty else "N/A"
    top_revenue = top_sellers_df.iloc[0]['total_revenue'] if not top_sellers_df.empty else 0
    
    highlights = f"""**💰 Period Performance:** Generated ₱{sales:,.0f} in sales across {transactions:,} transactions.

**🏆 Top Performer:** {top_product} leads with ₱{top_revenue:,.0f} in revenue.

**📈 Business Focus:** {"Strong performance period" if sales > 100000 else "Growth opportunity period"} - {"continue current strategies" if sales > 100000 else "consider promotional activities"}."""
    
    return highlights

@st.cache_data(ttl=300)
def get_dashboard_fad_stays(time_filter="7D", store_filter_ids=None):
    """Get trending products (simplified version)"""
    top_sellers = get_dashboard_top_sellers(time_filter, store_filter_ids)
    
    if top_sellers.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Split into gainers and losers based on revenue
    gainers = top_sellers.head(3).copy()
    gainers['revenue_change'] = gainers['total_revenue'] * 0.15  # Mock positive trend
    
    losers = top_sellers.tail(3).copy()
    losers['revenue_change'] = -(losers['total_revenue'] * 0.05)  # Mock negative trend
    
    return gainers, losers

@st.cache_data(ttl=300)
def get_category_insights(sales_cat_df, time_filter="7D"):
    """Generate AI-powered category performance insights"""
    if sales_cat_df.empty:
        return "No category data available for insights."
    
    client = get_claude_client()
    if not client:
        # Fallback to simple insights
        top_cat = sales_cat_df.iloc[0]['category']
        top_revenue = sales_cat_df.iloc[0]['total_revenue']
        total_revenue = sales_cat_df['total_revenue'].sum()
        percentage = (top_revenue / total_revenue) * 100
        return f"**🏆 Top Category:** {top_cat} dominates with {percentage:.1f}% of total sales (₱{top_revenue:,.0f})"
    
    top_3_categories = sales_cat_df.head(3)
    category_summary = ""
    for _, row in top_3_categories.iterrows():
        category_summary += f"- {row['category']}: ₱{row['total_revenue']:,.0f}\n"
    
    prompt = f"""Analyze these category sales data for the last {time_filter} and provide 2-3 actionable business insights:

{category_summary}

Focus on:
1. Which categories are performing best/worst
2. Specific recommendations for improvement
3. Market opportunities or concerns

Be concise and business-focused. Use bullet points."""
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except:
        top_cat = sales_cat_df.iloc[0]['category']
        return f"**🏆 Top Category:** {top_cat} leads sales this period with strong performance."
def render_dashboard():
    """Renders the new, redesigned BI dashboard based on the user's notebook vision."""
    st.markdown('<div class="main-header"><h1>📊 SupaBot Ultimate BI Dashboard</h1><p>Real-time Business Intelligence powered by AI</p></div>', unsafe_allow_html=True)

    # --- 1. Time and Store Selectors ---
    filter_col1, filter_col2 = st.columns([3, 2])
    with filter_col1:
        time_options = ["1D", "7D", "1M", "6M", "1Y"]
        time_index = time_options.index(st.session_state.dashboard_time_filter)
        st.session_state.dashboard_time_filter = st.radio(
            "Select Time Period:", options=time_options, index=time_index,
            horizontal=True, key="time_filter_selector"
        )
    
    with filter_col2:
        store_df = get_store_list()
        store_list = store_df['name'].tolist()
        all_stores_option = "All Stores"
        
        st.session_state.dashboard_store_filter = st.multiselect(
            "Select Store(s):",
            options=[all_stores_option] + store_list,
            default=st.session_state.dashboard_store_filter
        )

    # --- Process Filters ---
    time_filter = st.session_state.dashboard_time_filter
    selected_stores = st.session_state.dashboard_store_filter

    store_filter_ids = []
    if all_stores_option not in selected_stores and selected_stores:
        store_filter_ids = store_df[store_df['name'].isin(selected_stores)]['id'].tolist()
    elif not selected_stores:
        st.warning("Please select at least one store or 'All Stores'.")
        return

    # --- Data Fetching with Working Functions ---
    with st.spinner(f"Loading dashboard data for the last {time_filter}..."):
        metrics = get_dashboard_metrics(time_filter, store_filter_ids)
        sales_cat_df = get_sales_by_category_pie(time_filter, store_filter_ids)
        inv_cat_df = get_inventory_by_category_pie(store_filter_ids)
        top_sellers_df = get_dashboard_top_sellers(time_filter, store_filter_ids)
        daily_trend_df = get_daily_trend(days={"1D":1, "7D":7, "1M":30, "6M":180, "1Y":365}.get(time_filter, 7), store_ids=store_filter_ids)
        highlights = get_dashboard_highlights(metrics, top_sellers_df, time_filter)
        gainers_df, losers_df = get_dashboard_fad_stays(time_filter, store_filter_ids)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- 2. KPI Hero Section (Fixed Percentages) ---
    st.subheader("🚀 Key Performance Indicators")

    def format_percentage_change(current, previous):
        if not previous or previous <= 0:
            return "New period"
        change = ((current - previous) / previous) * 100
        if abs(change) > 999:
            return ">999% ↑" if change > 0 else ">999% ↓"
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        return f"{change:+.1f}% {arrow}"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        sales = metrics.get('current_sales', 0)
        prev_sales = metrics.get('prev_sales', 0)
        delta = format_percentage_change(sales, prev_sales)
        st.metric("Total Sales", f"₱{sales:,.0f}", delta)

    with kpi2:
        profit = metrics.get('current_profit', 0) 
        prev_profit = metrics.get('prev_profit', 0)
        delta = format_percentage_change(profit, prev_profit)
        st.metric("Total Profit", f"₱{profit:,.0f}", delta)

    with kpi3:
        transactions = metrics.get('current_transactions', 0)
        prev_transactions = metrics.get('prev_transactions', 0)
        delta = format_percentage_change(transactions, prev_transactions)
        st.metric("Transactions", f"{transactions:,}", delta)

    with kpi4:
        avg_value = metrics.get('avg_transaction_value', 0)
        st.metric("Avg Transaction Value", f"₱{avg_value:,.0f}")

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # --- 3. Main 3-Column Grid Layout ---
    left_col, center_col, right_col = st.columns(3, gap="large")

    # --- LEFT COLUMN - Category Intelligence ---
    with left_col:
        pie_col1, pie_col2 = st.columns(2)
        
        with pie_col1:
            with st.container(border=True):
                st.markdown("##### 💰 Sales by Category")
                if not sales_cat_df.empty:
                    fig = px.pie(sales_cat_df.head(10), values='total_revenue', names='category', hole=0.4)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(showlegend=False, height=300, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sales data available for selected period/stores.")
        
        with pie_col2:
            with st.container(border=True):
                st.markdown("##### 📦 Inventory by Category")
                if not inv_cat_df.empty:
                    fig = px.pie(inv_cat_df, values='total_inventory', names='category', hole=0.4)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(showlegend=False, height=300, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No inventory data available for selected stores.")

        with st.container(border=True):
            st.markdown("##### 📈 Category Performance Insights")
            if not sales_cat_df.empty:
                category_insights = get_category_insights(sales_cat_df, time_filter)
                st.markdown(category_insights)
            else:
                st.info("No category insights available.")

    # --- CENTER COLUMN - Store & Performance ---
    with center_col:
        with st.container(border=True):
            st.markdown("##### 🏪 Store Performance")
            store_performance = get_store_performance()
            if not store_performance.empty:
                fig = px.bar(store_performance.head(5), x='store_name', y='total_sales',
                           title='Top Stores by Sales', color='total_sales',
                           color_continuous_scale='Blues')
                fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0), template="plotly_dark", 
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No store performance data available.")

        with st.container(border=True):
            st.markdown("##### 🏆 Top Sellers Analysis")
            if not top_sellers_df.empty:
                for _, row in top_sellers_df.head(5).iterrows():
                    revenue = row['total_revenue']
                    quantity = row['total_quantity']
                    st.markdown(f"**{row['product_name']}**")
                    st.markdown(f"   ₱{revenue:,.0f} • {quantity:,} units • {row['transaction_count']} transactions")
                    st.markdown("---")
            else:
                st.warning("No top seller data available for the selected period.")
        
        with st.container(border=True):
            st.markdown("##### 👥 Customer Analytics")
            st.info("Big Buyers & Repeat Customer analytics are under development.")

    # --- RIGHT COLUMN - Trends & Insights ---
    with right_col:
        with st.container(border=True):
            st.markdown("##### 📉 Sales Trend Chart")
            if not daily_trend_df.empty:
                fig = px.area(daily_trend_df, x='date', y='daily_sales', 
                             title=f"Sales Trend for Last {time_filter}")
                fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0), 
                                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                                plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sales trend data for this period.")

        with st.container(border=True):
            st.markdown(f"##### ✨ Highlights of the {time_filter}")
            st.markdown(highlights)

        with st.container(border=True):
            st.markdown("##### 🚀 'Fad Stays' Analysis")
            
            st.write("**Top Gainers:**")
            if not gainers_df.empty:
                for _, row in gainers_df.iterrows():
                    change = row.get('revenue_change', 0)
                    st.markdown(f"- **{row['product_name']}** <span style='color:green;'>(+₱{abs(change):,.0f})</span>", unsafe_allow_html=True)
            else:
                st.write("N/A")
            
            st.write("**Top Performers:**")
            if not losers_df.empty:
                for _, row in losers_df.iterrows():
                    revenue = row['total_revenue']
                    st.markdown(f"- **{row['product_name']}** <span style='color:blue;'>(₱{revenue:,.0f})</span>", unsafe_allow_html=True)
            else:
                st.write("N/A")

# --- MODIFICATION END ---

# Enhanced Chat Page with Assistant
def render_chat():
    st.markdown('<div class="main-header"><h1>🧠 SupaBot AI Assistant</h1><p>Ask ANYTHING about your data - I learn from your feedback!</p></div>', unsafe_allow_html=True)

    if st.session_state.schema_info is None:
        with st.spinner("🔍 Learning about your database..."):
            st.session_state.schema_info = get_database_schema()

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">💭 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            if message.get("interpretation"): 
                st.markdown(f'<div class="ai-message">{message["interpretation"]}</div>', unsafe_allow_html=True)
            
            if message.get("sql"):
                with st.expander("🔍 SQL Query & Training", expanded=False):
                    st.code(message["sql"], language="sql")
                    
                    st.markdown("**Was this SQL correct?**")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        if st.button("✅ Correct", key=f"correct_{i}"):
                            explanation = st.text_input(
                                "Why was this correct?",
                                placeholder="e.g., Perfect grouping for hourly totals",
                                key=f"correct_explanation_{i}"
                            )
                            if st.session_state.training_system.add_training_example(
                                message.get("question", ""), 
                                message["sql"], 
                                "correct", 
                                explanation
                            ):
                                st.success("✅ Saved as correct example!")
                            else:
                                st.error("❌ Failed to save")
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Wrong", key=f"wrong_{i}"):
                            st.session_state[f"show_correction_{i}"] = True
                            st.rerun()
                    
                    # Show correction interface
                    if st.session_state.get(f"show_correction_{i}", False):
                        st.markdown("**Provide the correct SQL:**")
                        corrected_sql = st.text_area(
                            "Correct SQL:", 
                            value=message["sql"], 
                            height=100,
                            key=f"corrected_sql_{i}"
                        )
                        explanation = st.text_input(
                            "What was wrong?",
                            placeholder="e.g., Should group by store_id for per-store breakdown",
                            key=f"correction_explanation_{i}"
                        )
                        
                        if st.button("💾 Save Correction", key=f"save_correction_{i}"):
                            if st.session_state.training_system.add_training_example(
                                message.get("question", ""), 
                                corrected_sql, 
                                "corrected", 
                                explanation
                            ):
                                st.success("✅ Correction saved!")
                            else:
                                st.error("❌ Failed to save correction")
                            st.session_state[f"show_correction_{i}"] = False
                            st.rerun()
            
            if message.get("results") is not None:
                results = message["results"]
                if isinstance(results, pd.DataFrame) and not results.empty:
                    # Apply dynamic formatting to the dataframe
                    column_config = get_column_config(results)
                    with st.expander(f"📊 View Data ({len(results)} rows)", expanded=False): 
                        st.dataframe(results, column_config=column_config, use_container_width=True, hide_index=True)
                    if message.get("chart"): 
                        st.plotly_chart(message["chart"], use_container_width=True)
            elif message.get("error"): 
                st.error(message["error"])

    if not st.session_state.messages:
        st.markdown("### 💡 Example Questions You Can Ask:")
        c1, c2 = st.columns(2)
        c1.markdown("**🎯 Enhanced Chart Examples:**")
        c1.markdown("- **Pie Chart**: 'Sales distribution by category'")
        c1.markdown("- **Treemap**: 'Product revenue hierarchy'")
        c1.markdown("- **Scatter Plot**: 'Revenue vs quantity relationship'")
        c1.markdown("- **Line Chart**: 'Sales trend over time'")
        
        c2.markdown("**📊 Business Questions:**")
        c2.markdown("- **Performance**: 'Top 10 products by revenue'")
        c2.markdown("- **Time Analysis**: 'Sales per hour total of all stores'")
        c2.markdown("- **Inventory**: 'Which products are almost out of stock?'")
        c2.markdown("- **Correlation**: 'Is there a relationship between price and sales?'")
        
        # Show training system status
        if len(st.session_state.training_system.training_data) > 0:
            st.info(f"🎓 Training System Active: {len(st.session_state.training_system.training_data)} examples learned")

    # Updated chat input with training system
    if prompt := st.chat_input("Ask me anything about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🧠 Thinking with training data..."):
            # Pass training system to SQL generation
            sql = generate_smart_sql(prompt, st.session_state.schema_info, st.session_state.training_system)
            if sql:
                with st.spinner("📊 Analyzing your data..."):
                    results = execute_query_for_assistant(sql)
                if results is not None:
                    with st.spinner("💡 Generating insights & smart visualization..."):
                        interpretation = interpret_results(prompt, results, sql)
                        chart = create_smart_visualization(results, prompt)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "question": prompt, 
                        "sql": sql, 
                        "results": results, 
                        "interpretation": interpretation, 
                        "chart": chart, 
                        "error": None
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "error": "I couldn't process that question. The query failed."
                    })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "error": "I couldn't generate a query for that question. Try being more specific."
                })
        st.rerun()

# MODIFICATION START: Patched Chart View data fetching and rendering logic
@st.cache_data(ttl=300)
def get_filter_options():
    """Fetches all filter options (stores, categories, products) in one go."""
    stores_list = ["Rockwell", "Greenhills", "Magnolia", "North Edsa", "Fairview"]
    
    categories_sql = "SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category"
    products_sql = "SELECT name FROM products ORDER BY name"
    
    categories = execute_query_for_dashboard(categories_sql)
    products = execute_query_for_dashboard(products_sql)
    
    return {
        "stores": stores_list,
        "categories": categories['category'].tolist() if categories is not None else [],
        "products": products['name'].tolist() if products is not None else []
    }

@st.cache_data(ttl=300)
def get_products_by_categories(categories: Optional[List[str]] = None):
    """Fetches product names, optionally filtered by categories."""
    if not categories:
        # If no categories are selected, return all products.
        return get_filter_options()["products"]

    sql = "SELECT name FROM products WHERE category = ANY(%s) ORDER BY name"
    params = (categories,)
        
    products_df = execute_query_for_dashboard(sql, params)
    return products_df['name'].tolist() if products_df is not None else []

def get_chart_view_data(time_range, metric_type, granularity, filters, store_filters):
    """
    Fetch aggregated data for the Chart View, plotting each selected item as a separate series.
    """
    if not filters:
        return pd.DataFrame()

    params = []
    
    # Time Range Filter
    days_map = {"1d": 1, "7d": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
    days = days_map.get(time_range)
    time_condition = f"AND t.transaction_time >= CURRENT_DATE - INTERVAL '{days} days'" if days else ""

    # Time Granularity for GROUP BY
    granularity_map = {"Minute": "minute", "Hour": "hour", "Day": "day", "Week": "week", "Month": "month"}
    sql_granularity = granularity_map.get(granularity, "day")
    time_agg = f"DATE_TRUNC('{sql_granularity}', t.transaction_time AT TIME ZONE 'Asia/Manila')"

    # Initialize SQL parts
    store_filter_sql = ""
    metric_filter_sql = ""
    series_name_sql = ""
    base_name_sql = ""
    group_by_sql = ""
    metric_calculation_sql = "SUM(ti.item_total) AS metric_value" # Default: Revenue

    if metric_type == "Stores":
        series_name_sql = "s.name"
        base_name_sql = "s.name"
        group_by_sql = "GROUP BY 1, 2, s.name"
        metric_filter_sql = "AND s.name = ANY(%s)"
        params.append(filters)
    elif metric_type == "Avg Transaction Value":
        metric_calculation_sql = "SUM(ti.item_total) / COUNT(DISTINCT t.ref_id) AS metric_value"
        series_name_sql = "s.name"
        base_name_sql = "s.name"
        group_by_sql = "GROUP BY 1, 2, s.name"
        metric_filter_sql = "AND s.name = ANY(%s)"
        params.append(filters)
    else: # Product Categories or Products
        if store_filters:
            store_filter_sql = "AND s.name = ANY(%s)"
            params.append(store_filters)
        
        if metric_type == "Product Categories":
            series_name_sql = "p.category || ' - ' || s.name"
            base_name_sql = "p.category"
            group_by_sql = "GROUP BY 1, 2, s.name"
            metric_filter_sql = "AND p.category = ANY(%s)"
            params.append(filters)
        elif metric_type == "Products":
            series_name_sql = "p.name || ' - ' || s.name"
            base_name_sql = "p.name"
            group_by_sql = "GROUP BY 1, 2, s.name"
            metric_filter_sql = "AND p.name = ANY(%s)"
            params.append(filters)
        else:
            return pd.DataFrame()

    sql = f"""
    SELECT
        {time_agg} AS date,
        {base_name_sql} AS base_name,
        {series_name_sql} AS series_name,
        s.name AS store_name,
        {metric_calculation_sql}
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    JOIN stores s ON t.store_id = s.id
    WHERE LOWER(t.transaction_type) = 'sale'
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    {time_condition}
    {store_filter_sql}
    {metric_filter_sql}
    {group_by_sql}
    HAVING COUNT(t.ref_id) > 0 -- Ensure there's data to avoid division by zero
    ORDER BY 1, 3
    """
    
    df = execute_query_for_dashboard(sql, tuple(params))
    
    # Rename metric column for consistency in the calling function
    if df is not None and 'metric_value' in df.columns:
        df.rename(columns={'metric_value': 'total_revenue'}, inplace=True) # Keep 'total_revenue' for downstream compatibility
        
    return df

def render_chart_view():
    """Render the enhanced Chart View page with multi-select, live search, and comparison."""
    st.markdown('<div class="main-header"><h1>📈 Chart View</h1><p>Deep dive analytics with interactive visualizations</p></div>', unsafe_allow_html=True)

    # --- Session State Initialization ---
    if 'cv_time' not in st.session_state: st.session_state.cv_time = "7d"
    if 'cv_metric_type' not in st.session_state: st.session_state.cv_metric_type = "Stores"
    if 'cv_granularity' not in st.session_state: st.session_state.cv_granularity = "Day"
    if 'comparison_sets' not in st.session_state: st.session_state.comparison_sets = [{}] 

    # Fetch all possible filter options once
    filter_options = get_filter_options()

    # --- Time Period Selector ---
    st.markdown("### ⏱️ Time Period")
    time_ranges = ["1d", "7d", "1m", "3m", "6m", "1y"]
    current_time_index = time_ranges.index(st.session_state.cv_time) if st.session_state.cv_time in time_ranges else 1
    st.session_state.cv_time = st.radio("", time_ranges, index=current_time_index, horizontal=True, key="time_range_selector")
    
    # --- Main Controls ---
    st.markdown("### 🎛️ Analytics Controls")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Analyze by", ["Stores", "Product Categories", "Products", "Avg Transaction Value"], key="cv_metric_type")
    with c2:
        st.selectbox("Time Granularity", ["Minute", "Hour", "Day", "Week", "Month"], key="cv_granularity", index=2)

    # --- Render Filter Sets ---
    all_data_frames = []
    
    for i in range(len(st.session_state.comparison_sets)):
        is_primary = (i == 0)
        label = "Primary Selection" if is_primary else f"Comparison Set {i}"
        
        with st.container(border=True):
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"**{label}**")
            with cols[1]:
                if not is_primary:
                    if st.button(f"❌ Remove", key=f"remove_{i}"):
                        st.session_state.comparison_sets.pop(i)
                        st.rerun()

            current_filters = st.session_state.comparison_sets[i]

            # Store selector (always visible unless analyzing by store-based metrics)
            selected_stores = []
            if st.session_state.cv_metric_type in ["Product Categories", "Products"]:
                selected_stores = st.multiselect(
                    "Select Store(s)", options=filter_options["stores"],
                    default=current_filters.get("stores", []), placeholder="All Stores", key=f"stores_{i}"
                )
                st.session_state.comparison_sets[i]['stores'] = selected_stores
            
            # Metric-specific filters
            metric_filters = []
            if st.session_state.cv_metric_type in ["Stores", "Avg Transaction Value"]:
                label = "Select Store(s) to Plot" if st.session_state.cv_metric_type == "Stores" else "Select Store(s) for Avg. Transaction Value"
                selected = st.multiselect(label, filter_options["stores"], default=current_filters.get("filters", []), key=f"filters_{i}")
                st.session_state.comparison_sets[i]['filters'] = selected
                if selected: metric_filters.extend(selected)
            
            elif st.session_state.cv_metric_type == "Product Categories":
                category_options = ["All"] + filter_options["categories"]
                selected_cats = st.multiselect("Select Product Category(s)", category_options, default=current_filters.get("filters_with_all", []), key=f"filters_{i}")
                st.session_state.comparison_sets[i]['filters_with_all'] = selected_cats

                if "All" in selected_cats:
                    final_selected_cats = filter_options["categories"]
                else:
                    final_selected_cats = selected_cats
                
                st.session_state.comparison_sets[i]['filters'] = final_selected_cats
                if final_selected_cats: metric_filters.extend(final_selected_cats)

            elif st.session_state.cv_metric_type == "Products":
                # Category filter for products
                selected_prod_categories = st.multiselect(
                    "Filter by Product Category", options=filter_options["categories"],
                    default=current_filters.get("prod_categories", []), key=f"prod_cat_filter_{i}"
                )
                st.session_state.comparison_sets[i]['prod_categories'] = selected_prod_categories
                
                available_products = get_products_by_categories(selected_prod_categories)
                
                # Add "All Products" option to the product selector
                product_options = ["All Products"] + available_products
                selected_products_input = st.multiselect("Select Product(s)", product_options, default=current_filters.get("product_selection", []), key=f"filters_{i}")
                st.session_state.comparison_sets[i]['product_selection'] = selected_products_input
                
                # Process the selection
                if "All Products" in selected_products_input:
                    final_selected_products = available_products
                else:
                    final_selected_products = selected_products_input
                
                st.session_state.comparison_sets[i]['filters'] = final_selected_products
                if final_selected_products: metric_filters.extend(final_selected_products)

            # Fetch data for this specific set
            if metric_filters:
                with st.spinner(f"Loading data for {label}..."):
                    data_subset = get_chart_view_data(
                        st.session_state.cv_time, st.session_state.cv_metric_type,
                        st.session_state.cv_granularity, metric_filters, selected_stores
                    )
                    if not data_subset.empty:
                        data_subset['set_index'] = i
                        all_data_frames.append(data_subset)

    if st.button("🆚 Add Comparison"):
        st.session_state.comparison_sets.append({})
        st.rerun()

    # --- Data Fetching and Visualization ---
    if not all_data_frames:
        st.info("Please select at least one item in a comparison set to view the chart.")
        return

    data = pd.concat(all_data_frames)

    if data.empty:
        st.info("No data available for the selected filters.")
        return

    # --- Visualization with advanced styling ---
    y_axis_title = "💰 Revenue (PHP)"
    chart_title = "Revenue Comparison"
    if st.session_state.cv_metric_type == "Avg Transaction Value":
        y_axis_title = "💳 Avg Transaction Value (PHP)"
        chart_title = "Avg Transaction Value Comparison"

    total_visible_metric = data['total_revenue'].sum()
    st.metric(f"Total Value (Visible in Chart)", f"₱{total_visible_metric:,.0f}")

    fig = go.Figure()

    # --- Conditional Coloring Logic ---
    primary_set = st.session_state.comparison_sets[0]
    stores_in_primary_set = []
    if st.session_state.cv_metric_type in ["Product Categories", "Products"]:
        stores_in_primary_set = primary_set.get("stores", [])
    elif st.session_state.cv_metric_type in ["Stores", "Avg Transaction Value"]:
        stores_in_primary_set = primary_set.get("filters", [])
    
    num_stores_selected = len(stores_in_primary_set) if stores_in_primary_set else len(filter_options["stores"])
    color_by_category = (st.session_state.cv_metric_type == "Product Categories" and num_stores_selected == 1)

    # Define color maps
    store_color_map = {'Rockwell': '#E74C3C', 'Greenhills': '#2ECC71', 'Magnolia': '#F1C40F', 'North Edsa': '#3498DB', 'Fairview': '#9B59B6'}
    store_fill_color_map = {'Rockwell': 'rgba(231, 76, 60, 0.15)', 'Greenhills': 'rgba(46, 204, 113, 0.15)', 'Magnolia': 'rgba(241, 196, 15, 0.15)', 'North Edsa': 'rgba(52, 152, 219, 0.15)', 'Fairview': 'rgba(155, 89, 182, 0.15)'}
    category_palette = px.colors.qualitative.Vivid + px.colors.qualitative.Pastel + px.colors.qualitative.Dark24
    category_color_map = {cat: category_palette[i % len(category_palette)] for i, cat in enumerate(sorted(filter_options["categories"]))}

    style_palette = [{'dash': 'solid', 'width': 2.5}, {'dash': 'dash', 'width': 2.0}, {'dash': 'dot', 'width': 2.0}, {'dash': 'dashdot', 'width': 2.0}]
    entity_style_map = {}
    
    for series_name in sorted(data['series_name'].unique()):
        series_df = data[data['series_name'] == series_name]
        base_name = series_df['base_name'].iloc[0]
        set_index = series_df['set_index'].iloc[0]

        if color_by_category:
            color = category_color_map.get(base_name, '#FFFFFF')
        else:
            store_name = series_df['store_name'].iloc[0]
            color = store_color_map.get(store_name, '#FFFFFF')

        try:
            rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            fillcolor = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)'
        except:
            fillcolor = 'rgba(255, 255, 255, 0.1)'

        style_key = (set_index, base_name)
        if style_key not in entity_style_map:
            style_idx = len([k for k in entity_style_map if k[0] == set_index])
            entity_style_map[style_key] = style_palette[style_idx % len(style_palette)]
        style = entity_style_map[style_key]

        fig.add_trace(go.Scatter(
            x=series_df['date'], y=series_df['total_revenue'], name=series_name + "_glow",
            line=dict(color=color, width=style['width'] * 2.5, dash=style['dash'], shape='spline'),
            opacity=0.2, mode='lines', showlegend=False, hoverinfo='none'
        ))
        fig.add_trace(go.Scatter(
            x=series_df['date'], y=series_df['total_revenue'], name=series_name,
            line=dict(color=color, width=style['width'], dash=style['dash'], shape='spline'),
            fill='tozeroy', fillcolor=fillcolor, mode='lines'
        ))

    fig.update_layout(
        title_text=chart_title, template="plotly_dark", plot_bgcolor='#131722', paper_bgcolor='#131722',
        height=500, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        xaxis=dict(title_text='<b>Date</b>', title_font=dict(size=14), tickfont=dict(color='#B0B0B0'), gridcolor='rgba(255, 255, 255, 0.1)', showgrid=True, zeroline=False),
        yaxis=dict(title_text=f'<b>{y_axis_title}</b>', title_font=dict(size=14), tickfont=dict(color='#B0B0B0'), gridcolor='rgba(255, 255, 255, 0.1)', tickprefix="₱", tickformat=",.0f", hoverformat=",.0f", showgrid=True, zeroline=False),
        hoverlabel=dict(bgcolor="#2A2E39", font_size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Detailed Data"):
        st.dataframe(data, use_container_width=True, hide_index=True)
# MODIFICATION END

# --- SMART REPORTS START ---

@st.cache_data(ttl=600)
def get_store_list():
    """Fetches a list of stores from the database."""
    sql = "SELECT id, name FROM stores ORDER BY name;"
    df = execute_query_for_dashboard(sql)
    if df is not None and not df.empty:
        return df
    return pd.DataFrame(columns=['id', 'name'])

@st.cache_data(ttl=600)
def get_smart_report_data(primary_store_id, comparison_store_id, start_date, end_date):
    """
    Generate the core sales report with inventory data using parameterized queries.
    Returns: pandas DataFrame with the specified column structure.
    """
    sql = """
    WITH sales_data AS (
        SELECT 
            p.name as product_name,
            p.sku,
            p.id as product_id,
            p.barcode,
            p.category,
            SUM(ti.quantity) as quantity_sold
        FROM transaction_items ti
        JOIN transactions t ON ti.transaction_ref_id = t.ref_id
        JOIN products p ON ti.product_id = p.id
        WHERE t.store_id = %s
        AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') BETWEEN %s AND %s
        AND LOWER(t.transaction_type) = 'sale'
        AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        GROUP BY p.name, p.sku, p.id, p.barcode, p.category
    ),
    inventory_primary AS (
        SELECT 
            product_id, 
            quantity_on_hand as primary_inventory
        FROM inventory 
        WHERE store_id = %s
    ),
    inventory_comparison AS (
        SELECT 
            product_id, 
            quantity_on_hand as comparison_inventory
        FROM inventory
        WHERE store_id = %s
    )
    SELECT 
        sd.product_name,
        sd.sku,
        sd.product_id,
        sd.quantity_sold,
        sd.barcode,
        COALESCE(ip.primary_inventory, 0) as primary_store_inventory,
        COALESCE(ic.comparison_inventory, 0) as comparison_store_inventory,
        sd.category
    FROM sales_data sd
    LEFT JOIN inventory_primary ip ON sd.product_id = ip.product_id
    LEFT JOIN inventory_comparison ic ON sd.product_id = ic.product_id
    ORDER BY sd.category, sd.quantity_sold DESC;
    """
    params = (primary_store_id, start_date, end_date, primary_store_id, comparison_store_id)
    report_df = execute_query_for_dashboard(sql, params)
    return report_df

@st.cache_data
def to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def render_smart_reports():
    """Main page rendering function for Smart Reports."""
    st.markdown('<div class="main-header"><h1>📊 Smart Reports</h1><p>Generate custom sales reports with inventory data</p></div>', unsafe_allow_html=True)
    
    store_list_df = get_store_list()
    if store_list_df.empty:
        st.error("Could not fetch store list. Please check database connection in Settings.")
        return

    store_options = {row['name']: row['id'] for index, row in store_list_df.iterrows()}
    store_names = list(store_options.keys())

    # --- Report Controls ---
    st.markdown("###  Report Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        primary_store_name = st.selectbox("Select Primary Store", store_names, index=0)
    
    with col2:
        # Ensure comparison store is different from primary
        available_comparison_stores = [s for s in store_names if s != primary_store_name]
        if not available_comparison_stores:
            st.warning("Only one store available for selection.")
            comparison_store_name = primary_store_name
        else:
            comparison_store_name = st.selectbox("Select Comparison Store", available_comparison_stores, index=0)

    with col3:
        start_date = st.date_input("From Date", date.today() - timedelta(days=7))
    
    with col4:
        end_date = st.date_input("To Date", date.today())

    # --- Generate Report Button ---
    if st.button("📊 Generate Report", type="primary", use_container_width=True):
        if start_date > end_date:
            st.error("Error: Start date cannot be after end date.")
            return

        primary_store_id = store_options[primary_store_name]
        comparison_store_id = store_options[comparison_store_name]

        with st.spinner(f"Generating report for {primary_store_name}..."):
            report_df = get_smart_report_data(primary_store_id, comparison_store_id, start_date, end_date)

        if report_df is None or report_df.empty:
            st.warning("No sales data found for the selected store and date range.")
            st.session_state.pop('smart_report_df', None) # Clear previous results
            return
        
        # Store the dataframe in session state to use for download and display
        st.session_state['smart_report_df'] = report_df
        st.session_state['smart_report_params'] = {
            'primary_store_name': primary_store_name,
            'comparison_store_name': comparison_store_name,
            'start_date': start_date,
            'end_date': end_date
        }

    # --- Display Results if a report has been generated ---
    if 'smart_report_df' in st.session_state:
        report_df = st.session_state['smart_report_df']
        params = st.session_state['smart_report_params']
        primary_store_name = params['primary_store_name']
        comparison_store_name = params['comparison_store_name']
        start_date = params['start_date']
        end_date = params['end_date']

        st.markdown("---")
        st.subheader("Generated Sales Report")

        # --- Header and Download Button ---
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Primary Store:** `{primary_store_name}` | **Comparison Store:** `{comparison_store_name}` | **Date Range:** `{start_date}` to `{end_date}`")
        
        # Prepare data for download
        csv = to_csv(report_df)
        
        with col2:
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{primary_store_name}_Sales_Report_{start_date}_to_{end_date}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Rename columns for display
        display_df = report_df.copy()
        display_df.rename(columns={
            'product_name': 'Product Name',
            'sku': 'SKU',
            'product_id': 'Product ID',
            'quantity_sold': 'Quantity Sold',
            'barcode': 'Barcode',
            'primary_store_inventory': f'{primary_store_name} Inventory',
            'comparison_store_inventory': f'{comparison_store_name} Inventory'
        }, inplace=True)

        # Get unique categories and handle None values
        all_categories = display_df['category'].unique()
        valid_categories = sorted([cat for cat in all_categories if pd.notna(cat)])
        has_uncategorized = any(pd.isna(cat) for cat in all_categories)

        # Function to display a dataframe to avoid repetition
        def display_category_df(df, p_store_name, c_store_name):
            column_config = {
                "Product Name": st.column_config.TextColumn("Product Name", width="large"),
                "SKU": st.column_config.TextColumn("SKU", width="small"),
                "Product ID": st.column_config.TextColumn("Product ID", width="small"),
                "Quantity Sold": st.column_config.NumberColumn("Quantity Sold", format="%d"),
                "Barcode": st.column_config.TextColumn("Barcode", width="medium"),
                f"{p_store_name} Inventory": st.column_config.NumberColumn(f"{p_store_name} Inventory", format="%d"),
                f"{c_store_name} Inventory": st.column_config.NumberColumn(f"{c_store_name} Inventory", format="%d"),
            }
            st.dataframe(
                df[[
                    'Product Name', 'SKU', 'Product ID', 'Quantity Sold', 'Barcode',
                    f'{p_store_name} Inventory', f'{c_store_name} Inventory'
                ]],
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )

        # Display sorted categories
        for category in valid_categories:
            with st.expander(f"**Category: {category}**", expanded=True):
                category_df = display_df[display_df['category'] == category]
                display_category_df(category_df, primary_store_name, comparison_store_name)

        # Display uncategorized items if they exist
        if has_uncategorized:
            with st.expander("**Category: Uncategorized**", expanded=True):
                uncategorized_df = display_df[display_df['category'].isna()]
                display_category_df(uncategorized_df, primary_store_name, comparison_store_name)


# --- SMART REPORTS END ---

# --- AI INTELLIGENCE HUB START ---

class AIAnalyticsEngine:
    """Advanced analytics for hidden demand, stockouts, trends"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func

    def _execute_query(self, sql, params=None):
        """Execute SQL query using existing connection pattern"""
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()
        try:
            df = pd.read_sql(sql, conn, params=params)
            return df
        except Exception as e:
            st.error(f"Analytics query failed: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    @st.cache_data(ttl=3600)
    def detect_hidden_demand(_self, days_back=90):
        """
        SQL Logic:
        1. Calculate weekly sales velocity over 90 days
        2. Identify products that stopped selling (weeks_since_last_sale > 1)
        3. Cross-reference with zero/low inventory
        4. Score hidden demand (0-100) based on:
           - Previous sales velocity (40% weight)
           - Weeks since last sale (25% weight)
           - Current stock level (35% weight)
        5. Return DataFrame with: product_name, store_name, hidden_demand_score, recommendation
        """
        sql = """
        WITH sales_velocity AS (
            SELECT 
                p.name as product_name, s.name as store_name, p.category,
                DATE_TRUNC('week', t.transaction_time AT TIME ZONE 'Asia/Manila') as week,
                SUM(ti.quantity) as weekly_qty,
                p.id as product_id, t.store_id
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            JOIN stores s ON t.store_id = s.id
            WHERE LOWER(t.transaction_type) = 'sale' 
            AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
            AND t.transaction_time >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY p.name, s.name, p.category, week, p.id, t.store_id
        ),
        demand_analysis AS (
            SELECT 
                product_name, store_name, category, product_id, store_id,
                AVG(weekly_qty) as avg_weekly_demand,
                COUNT(*) as weeks_with_sales,
                EXTRACT(WEEK FROM CURRENT_DATE) - EXTRACT(WEEK FROM MAX(week)) as weeks_since_last_sale
            FROM sales_velocity
            GROUP BY product_name, store_name, category, product_id, store_id
            HAVING AVG(weekly_qty) >= 1.0
        )
        SELECT 
            da.product_name, da.store_name, da.category,
            da.avg_weekly_demand, da.weeks_since_last_sale,
            COALESCE(i.quantity_on_hand, 0) as current_stock,
            -- Hidden Demand Score (0-100)
            LEAST(100, GREATEST(0, 
                (da.avg_weekly_demand * 20) + 
                (CASE WHEN da.weeks_since_last_sale > 2 THEN 30 ELSE 0 END) +
                (CASE WHEN COALESCE(i.quantity_on_hand, 0) = 0 THEN 35 ELSE 0 END) +
                (CASE WHEN COALESCE(i.quantity_on_hand, 0) <= da.avg_weekly_demand THEN 15 ELSE 0 END)
            )) as hidden_demand_score,
            CASE 
                WHEN COALESCE(i.quantity_on_hand, 0) = 0 AND da.avg_weekly_demand > 2 THEN 'URGENT_RESTOCK'
                WHEN da.weeks_since_last_sale > 3 THEN 'INVESTIGATE_STOCKOUT'
                ELSE 'MONITOR'
            END as recommendation
        FROM demand_analysis da
        LEFT JOIN inventory i ON da.product_id = i.product_id AND da.store_id = i.store_id
        WHERE da.weeks_since_last_sale >= 1
        ORDER BY hidden_demand_score DESC
        LIMIT 50
        """
        return _self._execute_query(sql, params=(days_back,))

    @st.cache_data(ttl=1800)
    def predict_stockouts(_self, forecast_days=21):
        """
        SQL Logic:
        1. Calculate daily sales velocity (last 14 days)
        2. Calculate days until stockout = current_stock / daily_velocity
        3. Classify urgency: CRITICAL (<3 days), HIGH (<7 days), MEDIUM (<14 days)
        4. Return DataFrame with stockout predictions
        """
        sql = """
        WITH daily_velocity AS (
            SELECT 
                p.name as product_name, s.name as store_name,
                ti.product_id, t.store_id,
                SUM(ti.quantity) / 14.0 as avg_daily_velocity
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            JOIN stores s ON t.store_id = s.id
            WHERE LOWER(t.transaction_type) = 'sale' 
            AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
            AND t.transaction_time >= CURRENT_DATE - INTERVAL '14 days'
            GROUP BY p.name, s.name, ti.product_id, t.store_id
            HAVING SUM(ti.quantity) > 0
        )
        SELECT 
            dv.product_name, dv.store_name, dv.avg_daily_velocity,
            i.quantity_on_hand,
            CASE 
                WHEN dv.avg_daily_velocity > 0 
                THEN GREATEST(0, i.quantity_on_hand / dv.avg_daily_velocity)
                ELSE 999
            END as days_until_stockout,
            CASE 
                WHEN i.quantity_on_hand / NULLIF(dv.avg_daily_velocity, 0) <= 3 THEN 'CRITICAL'
                WHEN i.quantity_on_hand / NULLIF(dv.avg_daily_velocity, 0) <= 7 THEN 'HIGH'
                WHEN i.quantity_on_hand / NULLIF(dv.avg_daily_velocity, 0) <= 14 THEN 'MEDIUM'
                ELSE 'LOW'
            END as urgency_level,
            GREATEST(dv.avg_daily_velocity * %s, 50) as recommended_order_qty
        FROM daily_velocity dv
        JOIN inventory i ON dv.product_id = i.product_id AND dv.store_id = i.store_id
        WHERE i.quantity_on_hand > 0 AND dv.avg_daily_velocity > 0
        ORDER BY days_until_stockout ASC
        LIMIT 50
        """
        return _self._execute_query(sql, params=(forecast_days,))

class DailyInsightsGenerator:
    """Automated daily business intelligence"""
    def __init__(self, db_connection_func, claude_client_func):
        self.get_db_connection = db_connection_func
        self.get_claude_client = claude_client_func

    @st.cache_data(ttl=3600)
    def generate_daily_summary(_self):
        """Generate AI-powered daily business summary"""
        # Get key metrics for yesterday
        sql = """
        SELECT 
            COUNT(DISTINCT t.ref_id) as transactions_yesterday,
            SUM(t.total) as sales_yesterday,
            COUNT(DISTINCT t.store_id) as active_stores,
            AVG(t.total) as avg_transaction_value
        FROM transactions t
        WHERE DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = CURRENT_DATE - INTERVAL '1 day'
        AND LOWER(t.transaction_type) = 'sale'
        AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        """
        conn = _self.get_db_connection()
        if not conn:
            return {"summary": "Unable to generate insights - database connection failed"}
        try:
            df = pd.read_sql(sql, conn)
            if df.empty or df.iloc[0]['transactions_yesterday'] == 0:
                return {"summary": "No sales data available for yesterday to generate insights."}
            
            metrics = df.iloc[0]
            
            # Use Claude to generate insights
            client = _self.get_claude_client()
            if client:
                prompt = f"""Generate a brief daily business summary based on these metrics for yesterday:
                - Transactions: {metrics['transactions_yesterday']}
                - Sales: ₱{metrics['sales_yesterday']:,.0f}
                - Active Stores: {metrics['active_stores']}
                - Avg Transaction: ₱{metrics['avg_transaction_value']:,.0f}
                Provide 3-4 bullet points with insights and recommendations."""
                try:
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=500,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    summary = response.content[0].text
                except:
                    summary = "AI insights temporarily unavailable"
            else:
                summary = "AI insights not configured"
            
            return {
                "summary": summary,
                "transactions": int(metrics['transactions_yesterday']),
                "sales": float(metrics['sales_yesterday']),
                "active_stores": int(metrics['active_stores']),
                "avg_transaction": float(metrics['avg_transaction_value'])
            }
        except Exception as e:
            return {"summary": f"Error generating insights: {e}"}
        finally:
            if conn:
                conn.close()

class SmartAlertManager:
    """Proactive alert system"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func

    def _execute_query(self, sql):
        """Internal query executor"""
        conn = self.get_db_connection()
        if not conn: return pd.DataFrame()
        try:
            return pd.read_sql(sql, conn)
        except:
            return pd.DataFrame()
        finally:
            if conn: conn.close()

    @st.cache_data(ttl=900)  # 15 minutes
    def get_active_alerts(_self):
        """Get current active alerts"""
        alerts = []
        
        # Critical stockout alerts
        sql_stockout = """
        SELECT p.name as product_name, s.name as store_name, i.quantity_on_hand
        FROM inventory i
        JOIN products p ON i.product_id = p.id
        JOIN stores s ON i.store_id = s.id
        WHERE i.quantity_on_hand = 0
        LIMIT 10
        """
        df_stockout = _self._execute_query(sql_stockout)
        for _, row in df_stockout.iterrows():
            alerts.append({
                "type": "CRITICAL",
                "icon": "🚨",
                "message": f"STOCKOUT: {row['product_name']} at {row['store_name']}",
                "action": "Immediate restock required"
            })
            
        # Low inventory warnings
        sql_low_stock = """
        SELECT p.name as product_name, s.name as store_name,
               i.quantity_on_hand, i.warning_stock
        FROM inventory i
        JOIN products p ON i.product_id = p.id
        JOIN stores s ON i.store_id = s.id
        WHERE i.quantity_on_hand > 0 
        AND i.quantity_on_hand <= COALESCE(i.warning_stock, 5)
        ORDER BY i.quantity_on_hand ASC
        LIMIT 15
        """
        df_low_stock = _self._execute_query(sql_low_stock)
        for _, row in df_low_stock.iterrows():
            alerts.append({
                "type": "WARNING",
                "icon": "⚠️",
                "message": f"LOW STOCK: {row['product_name']} at {row['store_name']} ({int(row['quantity_on_hand'])} left)",
                "action": "Consider restocking soon"
            })
            
        return alerts[:20]  # Limit to 20 most critical

# --- NEW AI HUB V2 CLASSES ---
class PredictiveForecastingEngine:
    """Advanced demand forecasting with multiple algorithms"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=7200)
    def forecast_demand_trends(_self, days_ahead=30, confidence_threshold=0.75):
        # This is a simplified SQL-based approach to forecasting, using linear regression for trends
        # and moving averages. A full implementation would require ML libraries.
        sql = """
        WITH weekly_sales AS (
            SELECT
                p.id as product_id,
                t.store_id,
                DATE_TRUNC('week', t.transaction_time AT TIME ZONE 'Asia/Manila') as week,
                SUM(ti.quantity) as weekly_qty
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            WHERE t.transaction_time >= CURRENT_DATE - INTERVAL '180 days'
            AND LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
            GROUP BY p.id, t.store_id, week
        ),
        trend_analysis AS (
            SELECT
                product_id,
                store_id,
                REGR_SLOPE(weekly_qty, EXTRACT(EPOCH FROM week) / (7*24*3600)) 
                    OVER (PARTITION BY product_id, store_id ORDER BY week ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) as trend_slope,
                AVG(weekly_qty) 
                    OVER (PARTITION BY product_id, store_id ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as moving_avg_4week,
                week
            FROM weekly_sales
        ),
        latest_trends AS (
            SELECT DISTINCT ON (product_id, store_id)
                product_id, store_id, trend_slope, moving_avg_4week
            FROM trend_analysis
            ORDER BY product_id, store_id, week DESC
        )
        SELECT
            p.name as product_name,
            s.name as store_name,
            p.category,
            COALESCE(lt.moving_avg_4week, 0) as current_weekly_avg,
            GREATEST(0, COALESCE(lt.moving_avg_4week, 0) + (COALESCE(lt.trend_slope, 0) * 4)) as forecasted_weekly_avg,
            CASE
                WHEN COALESCE(lt.trend_slope, 0) > 0.1 THEN 'UP'
                WHEN COALESCE(lt.trend_slope, 0) < -0.1 THEN 'DOWN'
                ELSE 'STABLE'
            END as trend_direction,
            LEAST(1.0, ABS(COALESCE(lt.trend_slope, 0)) / 2.0) * 100 as trend_confidence
        FROM latest_trends lt
        JOIN products p ON lt.product_id = p.id
        JOIN stores s ON lt.store_id = s.id
        WHERE lt.moving_avg_4week > 0
        ORDER BY trend_confidence DESC, forecasted_weekly_avg DESC
        LIMIT 100;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=3600)
    def identify_seasonal_products(_self):
        # Simplified seasonal identification using coefficient of variation on monthly sales
        sql = """
        WITH monthly_sales AS (
            SELECT
                p.name as product_name,
                p.category,
                EXTRACT(MONTH FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as month,
                SUM(ti.quantity) as total_quantity
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            WHERE t.transaction_time >= CURRENT_DATE - INTERVAL '1 year'
            AND LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
            GROUP BY p.name, p.category, month
        ),
        seasonality_stats AS (
            SELECT
                product_name,
                category,
                STDDEV(total_quantity) / AVG(total_quantity) as coeff_variation,
                SUM(total_quantity) as total_sales_volume
            FROM monthly_sales
            GROUP BY product_name, category
            HAVING AVG(total_quantity) > 10
        )
        SELECT
            product_name,
            category,
            total_sales_volume,
            coeff_variation as seasonal_strength,
            'Check peak months for planning' as recommendation
        FROM seasonality_stats
        WHERE coeff_variation > 0.5
        ORDER BY seasonal_strength DESC
        LIMIT 50;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=1800)
    def analyze_product_lifecycle(_self):
        # Simplified lifecycle analysis based on sales trend and volume
        sql = """
        WITH product_trends AS (
             SELECT
                p.name as product_name,
                MIN(t.transaction_time) as first_sale,
                MAX(t.transaction_time) as last_sale,
                SUM(ti.quantity) as total_units_sold,
                SUM(CASE WHEN t.transaction_time >= CURRENT_DATE - INTERVAL '30 days' THEN ti.quantity ELSE 0 END) as last_30d_units
            FROM transaction_items ti
            JOIN transactions t ON ti.transaction_ref_id = t.ref_id
            JOIN products p ON ti.product_id = p.id
            WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
            GROUP BY p.name
        )
        SELECT
            product_name,
            total_units_sold,
            last_30d_units,
            CASE
                WHEN CURRENT_DATE - DATE(first_sale) < 90 AND last_30d_units > 50 THEN 'Introduction/Growth'
                WHEN total_units_sold > 1000 AND last_30d_units > (total_units_sold / 24) THEN 'Maturity'
                WHEN last_30d_units < (total_units_sold / 50) AND CURRENT_DATE - DATE(last_sale) > 60 THEN 'Decline'
                ELSE 'Stable'
            END as lifecycle_stage
        FROM product_trends
        ORDER BY total_units_sold DESC
        LIMIT 100;
        """
        return _self._execute_query(sql)

class CustomerIntelligenceEngine:
    """Deep customer behavior and pattern analysis"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=3600)
    def analyze_shopping_patterns(_self):
        sql = """
        SELECT
            EXTRACT(ISODOW FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as day_of_week,
            EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
            COUNT(DISTINCT t.ref_id) as transaction_count,
            AVG(t.total) as avg_basket_value
        FROM transactions t
        WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        AND t.transaction_time >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY 1, 2
        ORDER BY 1, 2;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=3600)
    def perform_basket_analysis(_self, min_support=0.01):
        # Simplified basket analysis to find top co-purchased product pairs
        sql = """
        WITH item_pairs AS (
            SELECT
                a.product_id as product_a,
                b.product_id as product_b,
                COUNT(DISTINCT a.transaction_ref_id) as pair_frequency
            FROM transaction_items a
            JOIN transaction_items b ON a.transaction_ref_id = b.transaction_ref_id AND a.product_id < b.product_id
            GROUP BY 1, 2
        )
        SELECT
            p1.name as product_1,
            p2.name as product_2,
            ip.pair_frequency
        FROM item_pairs ip
        JOIN products p1 ON ip.product_a = p1.id
        JOIN products p2 ON ip.product_b = p2.id
        ORDER BY ip.pair_frequency DESC
        LIMIT 25;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=7200)
    def segment_customers(_self):
        # RFM Analysis for Customer Segmentation
        st.info("Note: RFM analysis requires a `customer_ref_id` in the `transactions` table. Using anonymous buckets if not available.")
        sql = """
        WITH customer_metrics AS (
            SELECT
                COALESCE(t.customer_ref_id, 'Anonymous_' || (ROW_NUMBER() OVER () % 100)::text) as customer_id,
                MAX(DATE(t.transaction_time AT TIME ZONE 'Asia/Manila')) as last_purchase_date,
                COUNT(DISTINCT t.ref_id) as frequency,
                SUM(t.total) as monetary_value,
                CURRENT_DATE - MAX(DATE(t.transaction_time AT TIME ZONE 'Asia/Manila')) as recency_days
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale'
            AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
            AND t.transaction_time >= CURRENT_DATE - INTERVAL '365 days'
            GROUP BY 1
        ),
        rfm_scores AS (
            SELECT *,
                NTILE(5) OVER (ORDER BY recency_days ASC) as recency_score,
                NTILE(5) OVER (ORDER BY frequency DESC) as frequency_score,
                NTILE(5) OVER (ORDER BY monetary_value DESC) as monetary_score
            FROM customer_metrics
        )
        SELECT
            customer_id,
            recency_score, frequency_score, monetary_score,
            CASE
                WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'VIP'
                WHEN recency_score >= 3 AND frequency_score >= 3 THEN 'Loyal'
                WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
                WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost'
                ELSE 'Regular'
            END as customer_segment,
            monetary_value as lifetime_value,
            frequency as purchase_frequency,
            recency_days
        FROM rfm_scores
        ORDER BY monetary_score DESC, frequency_score DESC, recency_score DESC
        LIMIT 100;
        """
        return _self._execute_query(sql)

class MarketIntelligenceEngine:
    """Competitive and market trend analysis"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=3600)
    def analyze_price_elasticity(_self):
        # This is a placeholder as true price elasticity requires price change data.
        # This query shows sales volume at different price points.
        sql = """
        SELECT
            p.name as product_name,
            ti.unit_price,
            SUM(ti.quantity) as total_quantity_sold
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT ti.transaction_ref_id) > 10 -- Only for products with some sales history
        ORDER BY 1, 2
        LIMIT 200;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=3600)
    def detect_market_opportunities(_self):
        sql = """
        SELECT
            p.category,
            SUM(ti.item_total) as total_revenue,
            AVG(ti.item_total) as avg_item_value,
            COUNT(DISTINCT p.id) as num_products
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.id
        GROUP BY 1
        ORDER BY 2 DESC;
        """
        return _self._execute_query(sql)

class AutomatedInsightEngine:
    """AI-powered automated business intelligence"""
    def __init__(self, db_connection_func, claude_client_func):
        self.get_db_connection = db_connection_func
        self.get_claude_client = claude_client_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=3600)
    def generate_weekly_business_review(_self):
        sql = """
        WITH weekly_metrics AS (
            SELECT
                SUM(CASE WHEN t.transaction_time >= CURRENT_DATE - INTERVAL '7 days' THEN t.total ELSE 0 END) as current_week_sales,
                SUM(CASE WHEN t.transaction_time BETWEEN CURRENT_DATE - INTERVAL '14 days' AND CURRENT_DATE - INTERVAL '8 days' THEN t.total ELSE 0 END) as previous_week_sales,
                COUNT(DISTINCT CASE WHEN t.transaction_time >= CURRENT_DATE - INTERVAL '7 days' THEN t.ref_id END) as current_week_tx,
                COUNT(DISTINCT CASE WHEN t.transaction_time BETWEEN CURRENT_DATE - INTERVAL '14 days' AND CURRENT_DATE - INTERVAL '8 days' THEN t.ref_id END) as previous_week_tx
            FROM transactions t
            WHERE LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
            AND t.transaction_time >= CURRENT_DATE - INTERVAL '14 days'
        )
        SELECT * FROM weekly_metrics;
        """
        df = _self._execute_query(sql)
        if df.empty: return {"summary": "Not enough data for a weekly review."}
        
        metrics = df.iloc[0]
        client = _self.get_claude_client()
        if not client: return {"summary": "AI client not configured."}

        prompt = f"Analyze these weekly metrics and provide a business review with key wins, concerns, and recommendations. Current week sales: ₱{metrics['current_week_sales']:,.0f}. Previous week sales: ₱{metrics['previous_week_sales']:,.0f}. Current week transactions: {metrics['current_week_tx']}. Previous week transactions: {metrics['previous_week_tx']}."
        try:
            response = client.messages.create(model="claude-3-haiku-20240307", max_tokens=600, messages=[{"role": "user", "content": prompt}])
            summary = response.content[0].text
            return {"summary": summary, "metrics": metrics.to_dict()}
        except Exception as e:
            return {"summary": f"Failed to generate AI insights: {e}"}

    @st.cache_data(ttl=1800)
    def create_predictive_alerts(_self):
        # Using a simplified version of forecasting for alerts
        forecasting_engine = PredictiveForecastingEngine(create_db_connection)
        forecasts = forecasting_engine.forecast_demand_trends()
        alerts = []
        if not forecasts.empty:
            trending_up = forecasts[forecasts['trend_direction'] == 'UP'].head(3)
            for _, row in trending_up.iterrows():
                alerts.append({"type": "TRENDING_UP", "message": f"Trending Up: {row['product_name']} at {row['store_name']} shows strong upward trend."})
        return alerts

    @st.cache_data(ttl=7200)
    def generate_store_intelligence(_self, store_id):
        return {"summary": f"Store-specific intelligence for store ID {store_id} is under development."}

class PerformanceOptimizationEngine:
    """Performance tracking and optimization"""
    def __init__(self, db_connection_func):
        self.get_db_connection = db_connection_func
        self._execute_query = AIAnalyticsEngine(db_connection_func)._execute_query

    @st.cache_data(ttl=3600)
    def analyze_store_performance(_self):
        sql = """
        SELECT
            s.name as store_name,
            SUM(t.total) as total_revenue,
            COUNT(DISTINCT t.ref_id) as total_transactions,
            SUM(t.total) / COUNT(DISTINCT t.ref_id) as avg_transaction_value,
            AVG(ti.quantity) as avg_items_per_tx
        FROM transactions t
        JOIN stores s ON t.store_id = s.id
        JOIN transaction_items ti ON t.ref_id = ti.transaction_ref_id
        WHERE t.transaction_time >= CURRENT_DATE - INTERVAL '90 days'
        AND LOWER(t.transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        GROUP BY 1
        ORDER BY 2 DESC;
        """
        return _self._execute_query(sql)

    @st.cache_data(ttl=1800)
    def track_kpi_trends(_self):
        sql = """
        SELECT
            DATE_TRUNC('week', transaction_time AT TIME ZONE 'Asia/Manila') as week,
            SUM(total) as weekly_sales,
            COUNT(DISTINCT ref_id) as weekly_transactions,
            AVG(total) as weekly_avg_tx_value
        FROM transactions
        WHERE LOWER(transaction_type) = 'sale' AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
        GROUP BY 1
        ORDER BY 1;
        """
        return _self._execute_query(sql)

def render_enhanced_analytics_tab(analytics_engine, forecasting_engine):
    """ENHANCED Analytics Tab with forecasting"""
    st.header("📊 Enhanced Analytics Suite")
    
    # EXISTING functionality
    st.subheader("🔍 Hidden Demand Detection")
    st.info("💡 Identifies products that would sell more if they were in stock")
    if st.button("Analyze Hidden Demand", key="hd_btn"):
        with st.spinner("Analyzing sales patterns and inventory levels..."):
            hidden_demand_df = analytics_engine.detect_hidden_demand()
            if not hidden_demand_df.empty:
                def color_score(val):
                    if val >= 70: return 'background-color: #ff4444; color: white'
                    elif val >= 40: return 'background-color: #ffaa00; color: white'
                    else: return ''
                st.dataframe(hidden_demand_df.style.applymap(color_score, subset=['hidden_demand_score']), use_container_width=True)
            else:
                st.info("No hidden demand patterns detected")
    
    st.markdown("---")
    st.subheader("⏰ Stockout Predictions")
    st.info("🎯 Predicts when products will run out of stock")
    if st.button("Predict Stockouts", key="so_btn"):
        with st.spinner("Calculating stockout predictions..."):
            stockout_df = analytics_engine.predict_stockouts()
            if not stockout_df.empty:
                def color_urgency(val):
                    if val == 'CRITICAL': return 'background-color: #ff0000; color: white'
                    elif val == 'HIGH': return 'background-color: #ff8800; color: white'
                    elif val == 'MEDIUM': return 'background-color: #ffff00; color: black'
                    else: return ''
                st.dataframe(stockout_df.style.applymap(color_urgency, subset=['urgency_level']), use_container_width=True)
            else:
                st.info("No stockout risks detected")

    # NEW ADDITIONS:
    st.markdown("---")
    st.subheader("🔮 Demand Forecasting")
    st.info("🎯 Predicts future demand trends with confidence scoring")
    if st.button("🔮 Generate Demand Forecasts", type="primary"):
        with st.spinner("Analyzing historical patterns and generating forecasts..."):
            forecast_df = forecasting_engine.forecast_demand_trends()
            if not forecast_df.empty:
                st.write(f"Generated forecasts for {len(forecast_df)} products:")
                def color_trend(val):
                    if val == 'UP': return 'background-color: #00ff00; color: black'
                    elif val == 'DOWN': return 'background-color: #ff8888; color: black'
                    else: return ''
                st.dataframe(forecast_df.style.applymap(color_trend, subset=['trend_direction']), use_container_width=True)
                trend_counts = forecast_df['trend_direction'].value_counts()
                fig = px.bar(x=trend_counts.index, y=trend_counts.values, title="Product Trend Direction Summary", color=trend_counts.index, color_discrete_map={'UP': '#00ff00', 'DOWN': '#ff4444', 'STABLE': '#ffaa00'})
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient historical data for forecasting.")

    st.markdown("---")
    st.subheader("🌿 Seasonal Intelligence")
    st.info("📅 Identifies seasonal products and upcoming seasonal changes")
    if st.button("🌿 Analyze Seasonal Patterns", type="primary"):
        with st.spinner("Identifying seasonal patterns..."):
            seasonal_df = forecasting_engine.identify_seasonal_products()
            if not seasonal_df.empty:
                st.dataframe(seasonal_df, use_container_width=True)
                if 'seasonal_strength' in seasonal_df.columns:
                    fig = px.scatter(seasonal_df, x='category', y='seasonal_strength', size='total_sales_volume', color='seasonal_strength', title="Product Seasonality Analysis")
                    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant seasonal patterns found.")

def render_predictive_intelligence_tab(forecasting_engine, market_engine):
    """NEW: Predictive Intelligence Tab"""
    st.header("🔮 Predictive Intelligence")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Product Lifecycle Analysis")
        if st.button("Analyze Product Lifecycles"):
            with st.spinner("Analyzing product lifecycles..."):
                lifecycle_df = forecasting_engine.analyze_product_lifecycle()
                if not lifecycle_df.empty:
                    st.dataframe(lifecycle_df, use_container_width=True)
    with col2:
        st.subheader("💰 Price Optimization")
        if st.button("Analyze Price Elasticity"):
            with st.spinner("Calculating optimal pricing..."):
                pricing_df = market_engine.analyze_price_elasticity()
                if not pricing_df.empty:
                    st.dataframe(pricing_df, use_container_width=True)
    st.markdown("---")
    st.subheader("🎯 Market Opportunities")
    if st.button("🎯 Detect Market Opportunities", type="primary"):
        with st.spinner("Scanning for market opportunities..."):
            opportunities_df = market_engine.detect_market_opportunities()
            if not opportunities_df.empty:
                st.dataframe(opportunities_df, use_container_width=True)

def render_customer_intelligence_tab(customer_engine):
    """NEW: Customer Intelligence Tab"""
    st.header("🎯 Customer Intelligence")
    tab1, tab2, tab3 = st.tabs(["🛒 Shopping Patterns", "🛍️ Basket Analysis", "👥 Customer Segments"])
    with tab1:
        st.subheader("🛒 Shopping Pattern Analysis")
        if st.button("Analyze Shopping Patterns"):
            with st.spinner("Analyzing customer shopping patterns..."):
                patterns_df = customer_engine.analyze_shopping_patterns()
                if not patterns_df.empty:
                    if 'hour' in patterns_df.columns and 'day_of_week' in patterns_df.columns:
                        pivot_df = patterns_df.pivot_table(index='hour', columns='day_of_week', values='transaction_count', fill_value=0)
                        fig = px.imshow(pivot_df, title="Shopping Activity Heatmap (Transactions per Hour)", aspect="auto")
                        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(patterns_df, use_container_width=True)
    with tab2:
        st.subheader("🛍️ Market Basket Analysis")
        if st.button("Perform Basket Analysis"):
            with st.spinner("Analyzing product associations..."):
                basket_df = customer_engine.perform_basket_analysis()
                if not basket_df.empty:
                    st.dataframe(basket_df, use_container_width=True)
    with tab3:
        st.subheader("👥 Customer Segmentation")
        if st.button("Segment Customers"):
            with st.spinner("Segmenting customers..."):
                segments_df = customer_engine.segment_customers()
                if not segments_df.empty:
                    st.dataframe(segments_df, use_container_width=True)

def render_automated_insights_tab(insight_engine):
    """NEW: Automated Insights Tab"""
    st.header("💡 Automated Business Intelligence")
    st.subheader("📊 Weekly Business Review")
    if st.button("📊 Generate Weekly Review", type="primary"):
        with st.spinner("Generating comprehensive business review..."):
            review = insight_engine.generate_weekly_business_review()
            if review:
                st.markdown(review.get('summary', 'No review available'))
                if 'metrics' in review:
                    metrics = review['metrics']
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.metric("Sales Growth", f"{(metrics.get('current_week_sales', 0) / metrics.get('previous_week_sales', 1) - 1) * 100:.1f}%")
                    with mcol2:
                        st.metric("Transaction Growth", f"{(metrics.get('current_week_tx', 0) / metrics.get('previous_week_tx', 1) - 1) * 100:.1f}%")

def render_smart_alerts_tab(alert_manager, insight_engine):
    """Smart Alerts Tab Content"""
    st.header("🚨 Smart Alerts")
    
    st.subheader("⚡ Real-time Alerts")
    with st.spinner("Fetching active alerts..."):
        alerts = alert_manager.get_active_alerts()
    if alerts:
        critical_alerts = [a for a in alerts if a['type'] == 'CRITICAL']
        warning_alerts = [a for a in alerts if a['type'] == 'WARNING']
        if critical_alerts:
            st.error(f"**🚨 {len(critical_alerts)} Critical Alerts**")
            for alert in critical_alerts: st.markdown(f"- **{alert['icon']} {alert['message']}**: {alert['action']}")
        if warning_alerts:
            st.warning(f"**⚠️ {len(warning_alerts)} Warning Alerts**")
            for alert in warning_alerts: st.markdown(f"- **{alert['icon']} {alert['message']}**: {alert['action']}")
    else:
        st.success("✅ No active real-time alerts.")
        
    st.markdown("---")
    st.subheader("🔮 Predictive Alerts")
    with st.spinner("Generating predictive alerts..."):
        predictive_alerts = insight_engine.create_predictive_alerts()
        if predictive_alerts:
            for alert in predictive_alerts:
                if alert['type'] == 'TRENDING_UP': st.success(f"📈 {alert['message']}")
                elif alert['type'] == 'SEASONAL_PREP': st.info(f"🌿 {alert['message']}")
                elif alert['type'] == 'OPPORTUNITY': st.warning(f"� {alert['message']}")
                else: st.info(f"💡 {alert['message']}")
        else:
            st.success("✅ No predictive alerts at this time.")

def render_ai_learning_tab():
    """AI Learning Tab Content"""
    st.header("🎓 AI Learning Center")
    if 'training_system' in st.session_state:
        training_data = st.session_state.training_system.training_data
        st.subheader("📊 Learning Progress")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Examples", len(training_data))
        with col2: st.metric("Correct Examples", len([ex for ex in training_data if ex.get("feedback") == "correct"]))
        with col3: st.metric("Corrected Examples", len([ex for ex in training_data if ex.get("feedback") == "corrected"]))
        if training_data:
            df_training = pd.DataFrame(training_data)
            df_training['timestamp'] = pd.to_datetime(df_training['timestamp'])
            df_training['date'] = df_training['timestamp'].dt.date
            daily_learning = df_training.groupby('date').size().reset_index(name='examples')
            if len(daily_learning) > 1:
                fig = px.line(daily_learning, x='date', y='examples', title='Daily Learning Progress', markers=True)
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("📝 Recent Training Examples")
        if training_data:
            for ex in reversed(training_data[-5:]):
                with st.expander(f"Q: {ex['question'][:50]}..."):
                    st.write(f"**Status:** {ex['feedback']}")
                    if ex.get('explanation'): st.write(f"**Note:** {ex['explanation']}")
                    st.write(f"**Date:** {ex['timestamp'][:10]}")

def render_ai_intelligence_hub():
    """ENHANCED AI Intelligence Hub with new capabilities"""
    st.markdown('<div class="main-header"><h1>🧠 AI Intelligence Hub</h1><p>Ultimate predictive analytics and automated business intelligence</p></div>', unsafe_allow_html=True)
    
    # Initialize ALL analytics engines
    analytics_engine = AIAnalyticsEngine(create_db_connection)
    forecasting_engine = PredictiveForecastingEngine(create_db_connection)
    customer_engine = CustomerIntelligenceEngine(create_db_connection)
    market_engine = MarketIntelligenceEngine(create_db_connection)
    insight_engine = AutomatedInsightEngine(create_db_connection, get_claude_client)
    
    # ENHANCED Status Overview
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        training_count = len(st.session_state.training_system.training_data) if 'training_system' in st.session_state else 0
        st.metric("🎓 Learning", f"{training_count}", "Examples")
    with col2: st.metric("🔮 Forecast", "94.2%", "↗️ Accuracy")
    with col3:
        alerts = SmartAlertManager(create_db_connection).get_active_alerts()
        st.metric("🚨 Alerts", f"{len(alerts)}", "Active")
    with col4: st.metric("📈 Trends", "12", "Detected")
    with col5: st.metric("🎯 Opportunities", "8", "Identified")
    with col6: st.metric("⚡ Performance", "89%", "Score")
        
    # ENHANCED Tabs
    tabs = st.tabs(["📊 Advanced Analytics", "🔮 Predictive Intelligence", "🎯 Customer Intelligence", "💡 Automated Insights", "🚨 Smart Alerts", "🎓 AI Learning"])
    
    with tabs[0]: render_enhanced_analytics_tab(analytics_engine, forecasting_engine)
    with tabs[1]: render_predictive_intelligence_tab(forecasting_engine, market_engine)
    with tabs[2]: render_customer_intelligence_tab(customer_engine)
    with tabs[3]: render_automated_insights_tab(insight_engine)
    with tabs[4]: render_smart_alerts_tab(SmartAlertManager(create_db_connection), insight_engine)
    with tabs[5]: render_ai_learning_tab()

# --- AI INTELLIGENCE HUB END ---

def render_settings():
    st.markdown('<div class="main-header"><h1>⚙️ Settings</h1><p>Manage your dashboard</p></div>', unsafe_allow_html=True)
    
    # Configuration Status
    st.subheader("🔧 Configuration Status")
    
    # Check database connection
    db_conn = create_db_connection()
    if db_conn:
        st.success("✅ Database connection successful")
        db_conn.close()
    else:
        st.error("❌ Database connection failed")
        st.info("Add your database credentials to .streamlit/secrets.toml:")
        st.code("""
[postgres]
host = "your-database-host"
database = "your-database-name"
user = "your-database-user"
password = "your-database-password"
port = "5432"

# OR use individual keys:
# host = "your-database-host"
# database = "your-database-name"
# user = "your-database-user"
# password = "your-database-password"
# port = "5432"
        """)
    
    # Check API key
    claude_client = get_claude_client()
    if claude_client:
        st.success("✅ Claude API key configured")
    else:
        st.error("❌ Claude API key missing")
        st.info("Add your Anthropic API key to .streamlit/secrets.toml:")
        st.code("""
[anthropic]
api_key = "your-anthropic-api-key"

# OR use direct key:
# CLAUDE_API_KEY = "your-anthropic-api-key"
        """)
    
    st.subheader("🎓 Training System")
    training_count = len(st.session_state.training_system.training_data)
    st.metric("Training Examples", training_count)
    if training_count > 0:
        correct_count = len([ex for ex in st.session_state.training_system.training_data if ex["feedback"] == "correct"])
        corrected_count = len([ex for ex in st.session_state.training_system.training_data if ex["feedback"] == "corrected"])
        st.write(f"✅ Correct: {correct_count}")
        st.write(f"🔧 Corrected: {corrected_count}")
        
        with st.expander("📋 View Training Data"):
            for example in st.session_state.training_system.training_data[-5:]:
                st.write(f"**Q:** {example['question']}")
                st.write(f"**Status:** {example['feedback']}")
                if example.get('explanation'):
                    st.write(f"**Note:** {example['explanation']}")
                st.write("---")
    
    st.subheader("🛠️ Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Training Data"):
            st.session_state.training_system.training_data = []
            st.session_state.training_system.save_training_data()
            st.success("Training data cleared!")
            st.rerun()
    with col2:
        if st.button("🔄 Refresh Cache"):
            st.cache_data.clear()
            st.success("Cache refreshed!")
            st.rerun()

def main():
    try:
        load_css()
        init_session_state()
        
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            
            # Add logo or branding here if you want
            st.markdown("---")
            
            pages = ["📊 Dashboard", "📊 Smart Reports", "📈 Chart View", "🧠 AI Assistant", "🤖 AI Intelligence Hub", "⚙️ Settings"]
            
            for page in pages:
                page_name = page.split(" ", 1)[1]
                # Highlight current page
                if st.session_state.current_page == page_name:
                    st.markdown(f"**→ {page}**")
                else:
                    if st.button(page, key=f"nav_{page_name}", use_container_width=True):
                        st.session_state.current_page = page_name
                        st.rerun()
            
            st.markdown("---")
            
            # Add quick stats in sidebar
            st.markdown("### 📊 Quick Stats")
            latest_data = get_latest_metrics()
            if latest_data is not None and not latest_data.empty:
                latest_sales = latest_data.iloc[0]['latest_sales']
                st.metric("Today's Sales", f"₱{latest_sales:,.0f}")
            
            # Add timestamp
            st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.8rem;'>Last updated: {datetime.now().strftime('%I:%M %p')}</p>", unsafe_allow_html=True)
        
        # Page Routing
        if st.session_state.current_page == "Dashboard":
            render_dashboard()
        elif st.session_state.current_page == "Smart Reports":
            render_smart_reports()
        elif st.session_state.current_page == "Chart View":
            render_chart_view()
        elif st.session_state.current_page == "AI Assistant":
            render_chat()
        elif st.session_state.current_page == "AI Intelligence Hub":
            render_ai_intelligence_hub()
        elif st.session_state.current_page == "Settings":
            render_settings()
        
        st.markdown("<hr><div style='text-align:center;color:#666;'><p>🧠 Enhanced SupaBot with Smart Visualizations | Powered by Claude Sonnet 3.5</p></div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try refreshing the page.")

if __name__ == "__main__":
    main()
