# =========================================================================
# CONFIGURATION & IMPORTS
# =========================================================================
import streamlit as st
st.set_page_config(
    page_title="🛒 E-Commerce Analytics, Churn & LTV Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
import warnings
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import ssl
from io import BytesIO
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer)
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, classification_report)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import re
import unicodedata
import pycountry
import random
import string
from datetime import timedelta
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# =========================================================================
# CONFIGURATION
# =========================================================================
GOOGLE_DRIVE_SHARE_URL = "https://drive.google.com/file/d/1BlVZUbcUUVuYTZXYHfy0kBaN8Y83VGN8/view"
LOCAL_DB_PATH = "ecommerce_raw.db"

# Email Configuration (Set these as environment variables for security)
EMAIL_CONFIG = {
    "SMTP_SERVER": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "SMTP_PORT": int(os.getenv("SMTP_PORT", "587")),
    "SENDER_EMAIL": os.getenv("SENDER_EMAIL", ""),
    "SENDER_PASSWORD": os.getenv("SENDER_PASSWORD", ""),
}

# =========================================================================
# ROBUST DATABASE DOWNLOAD
# =========================================================================

def convert_to_direct_download(url: str) -> str:
    """Converts Google Drive share URL to direct download URL."""
    if "drive.google.com" in url:
        if "/uc?export=download" in url:
            return url
        elif "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

@st.cache_resource(show_spinner=False)
def download_and_validate_database(url: str, local_path: str) -> str:
    """
    Downloads database from external URL with automatic Google Drive URL conversion.
    Includes quota handling and validation.
    """
    direct_url = convert_to_direct_download(url)
    
    # Check if valid file already exists
    if os.path.exists(local_path) and _is_valid_sqlite_db(local_path):
        st.sidebar.success("✅ Using cached database")
        return local_path
    
    st.info("🔽 Downloading database for first run...")
    
    try:
        file_bytes = _download_with_google_drive_support(direct_url)
        
        # Save to file
        with open(local_path, 'wb') as f:
            f.write(file_bytes)
        
        # Validate
        if not _is_valid_sqlite_db(local_path):
            if os.path.exists(local_path):
                os.remove(local_path)
            
            st.error("❌ Downloaded file is NOT a valid SQLite database!")
            st.error("The URL is not providing the raw file.")
            st.markdown("### 🔧 Troubleshooting Steps:")
            st.markdown("""
            1. **Check Sharing Settings:**
               - Right-click file in Google Drive → Share
               - Change to 'Anyone with the link' (not 'Restricted')
               
            2. **Verify URL Format:**
               - Your share URL should be: `https://drive.google.com/file/d/.../view`
               - The app will auto-convert it to direct download
               
            3. **Test the URL in a browser:**
               - Use the direct URL in a private/incognito browser
               - It should start downloading immediately (not show a preview)
            """)
            st.code(direct_url)
            st.stop()
        
        st.success("✅ Database downloaded and validated!")
        return local_path
        
    except Exception as e:
        st.error(f"❌ Failed to download database: {str(e)}")
        st.stop()

def _download_with_google_drive_support(url: str) -> bytes:
    """
    Downloads file, handling Google Drive's virus scan warnings and quotas.
    """
    session = requests.Session()
    response = session.get(url, stream=True, timeout=60)
    
    # Check for Google Drive confirmation page
    if "drive.google.com" in url and "confirm=" not in url:
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_url = f"{url}&confirm={value}"
                response = session.get(confirm_url, stream=True, timeout=60)
                break
    
    response.raise_for_status()
    
    # Check content type
    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type:
        content_start = response.text[:5000]
        if "Google Drive - Quota exceeded" in content_start:
            raise Exception("Google Drive quota exceeded. File too popular or large.")
        elif "The file you have requested does not exist" in content_start:
            raise Exception("File not found. Check the file ID or sharing settings.")
        elif "drive.google.com" in url:
            st.error("Server returned HTML page instead of database file. Preview:")
            st.code(content_start[:500])
            raise Exception("Got HTML page instead of file. Check sharing settings.")
    
    # Download with progress
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = st.progress(0)
    downloaded = 0
    chunks = []
    
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                progress_bar.progress(min(downloaded / total_size, 1.0))
    
    progress_bar.empty()
    return b''.join(chunks)

def _is_valid_sqlite_db(filepath: str) -> bool:
    """
    Validates that a file is a proper SQLite database by checking header and connection.
    """
    try:
        if not os.path.exists(filepath):
            return False
        
        size = os.path.getsize(filepath)
        if size < 100:
            return False
        
        # Check SQLite header
        with open(filepath, 'rb') as f:
            header = f.read(16)
            if header != b'SQLite format 3\x00':
                return False
        
        # Test connection
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        cursor.fetchone()
        conn.close()
        return True
    except Exception:
        return False

# =========================================================================
# IN-MEMORY ETL PIPELINE
# =========================================================================

@st.cache_data(show_spinner=False)
def run_etl_pipeline(db_path: str, save_to_sqlite: bool = False):
    """
    Runs the ETL pipeline in-memory and returns cleaned DataFrames.
    """
    # Validate database before attempting to use it
    if not _is_valid_sqlite_db(db_path):
        st.error(f"❌ Invalid database file: {db_path}")
        st.error("The file is corrupted or not a SQLite database")
        st.stop()
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        available_tables = [row[0] for row in cursor.fetchall()]
        
        tables_to_load = ["customers", "orders", "products", "reviews"]
        tables_to_load = [t for t in tables_to_load if t in available_tables]
        
        if not tables_to_load:
            st.error(f"No required tables found. Available: {available_tables}")
            st.stop()
        
        # Extract raw data
        dataframes = {}
        for table in tables_to_load:
            df = pd.read_sql_query(f"SELECT * FROM [{table}]", conn)
            dataframes[table] = df
        
        # Transform (clean)
        print("\n" + "=" * 80)
        print("2. TRANSFORMING DATA...")
        print("=" * 80)

        cleaned_tables = {}

        # ----------------------------- CUSTOMERS TABLE -----------------------------
        if "customers" in dataframes:
            print("Cleaning customers table")
            df = dataframes["customers"].copy()

            # customer_id
            df["customer_id"] = df["customer_id"].astype(str).str.strip()
            df["customer_id"] = df["customer_id"].replace(["", "nan", "None", "NULL", "<NA>"], np.nan)
            df["customer_id"] = df["customer_id"].fillna("Unknown")

            # Name fields
            name_columns = ["first_name", "last_name"]
            for col in name_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace(["", "nan", "None", "NULL"], np.nan)
                    df[col] = df[col].str.title()
                    df[col] = df[col].fillna("Unknown")

            # Full name
            def build_full_name(row):
                first = row.get("first_name", "Unknown")
                last = row.get("last_name", "Unknown")
                if first != "Unknown" and last != "Unknown":
                    return f"{first} {last}"
                elif first != "Unknown":
                    return first
                elif last != "Unknown":
                    return last
                else:
                    return "Unknown"

            df["full_name"] = df.apply(build_full_name, axis=1)

            # COUNTRY
            if "country" in df.columns:
                df["country"] = df["country"].str.title().fillna("Unknown")

                def standardize_country(country):
                    if country == "Unknown":
                        return "Unknown"
                    try:
                        return pycountry.countries.lookup(country).name
                    except LookupError:
                        variants = {
                            "United States": "United States of America",
                            "USA": "United States of America",
                            "UK": "United Kingdom",
                            "U.K.": "United Kingdom",
                        }
                        return variants.get(country, country)

                df["country"] = df["country"].apply(standardize_country)

            # Dates
            date_cols = ["registration_date", "date_of_birth"]
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            if "last_login" in df.columns:
                df["last_login"] = pd.to_datetime(df["last_login"], errors="coerce")
                df["last_login_date"] = df["last_login"].dt.date
                df["last_login_time"] = df["last_login"].dt.time
                df = df.drop(columns=["last_login"], errors="ignore")

            if "account_age_days" in df.columns:
                df["account_age_days"] = df["account_age_days"].replace(
                    {"Unknown": np.nan, "unknown": np.nan, "": np.nan, pd.NA: np.nan}
                )
                df["account_age_days"] = pd.to_numeric(df["account_age_days"], errors="coerce")
                df["account_age_days"] = df["account_age_days"].abs()

                valid_vals = df["account_age_days"].dropna()
                if not valid_vals.empty:
                    rng = np.random.default_rng(seed=42)
                    min_val = valid_vals.min()
                    max_val = valid_vals.max()
                    nan_mask = df["account_age_days"].isna()
                    df.loc[nan_mask, "account_age_days"] = rng.integers(
                        low=int(min_val), high=int(max_val) + 1, size=nan_mask.sum()
                    )

            # Boolean columns
            boolean_columns = ["newsletter_subscribed", "marketing_consent"]
            boolean_map = {
                "true": True, "false": False, "yes": True, "no": False,
                "y": True, "n": False, "1": True, "0": False,
                True: True, False: False
            }
            for col in boolean_columns:
                if col in df.columns:
                    df[col] = (
                        df[col].astype(str).str.strip().str.lower()
                        .replace(["", "nan", "none", "null"], np.nan)
                        .map(boolean_map)
                    )

            # Credit tier
            if "credit_tier" in df.columns:
                df["credit_tier"] = df["credit_tier"].astype(str).str.strip().str.upper()

            # Preferred language
            if "preferred_language" in df.columns:
                language_map = {"ESPAÑOL": "ES", "SPANISH": "ES", "ENGLISH": "EN", "NONE": "EN"}
                df["preferred_language"] = (
                    df["preferred_language"].astype(str).str.strip().str.upper()
                    .replace(language_map)
                    .replace(["", "NAN", "NONE", "NULL"], np.nan)
                    .fillna("EN")
                )

            # Currency preference
            if "currency_preference" in df.columns:
                currency_map = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY"}
                df["currency_preference"] = (
                    df["currency_preference"].astype(str).str.strip().str.upper()
                    .replace(currency_map)
                    .replace(['', "", "NAN", "NONE", "NULL"], np.nan)
                )

            # Customer status
            if "customer_status" in df.columns:
                df["customer_status"] = (
                    df["customer_status"].astype(str).str.strip().str.title()
                    .replace(["", "Nan", "None", "Null"], np.nan)
                    .fillna("Unknown")
                )

            # Gender
            if "gender" in df.columns:
                gender_map = {
                    "m": "Male", "male": "Male",
                    "f": "Female", "female": "Female",
                    "other": "Other", "o": "Other"
                }
                df["gender"] = (
                    df["gender"].astype(str).str.strip().str.lower()
                    .replace(["", "nan", "none", "null"], np.nan)
                    .map(gender_map).fillna("Unknown")
                )

            # Numeric financial columns
            numeric_columns = ["total_spent", "avg_order_value", "loyalty_score"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Drop columns
            drop_columns = ["email", "phone", "address", "city", "state", "last_login", 
                           "first_name", "last_name", "address_line1", "last_login_time",
                           "last_login_date", "zip_code"]
            df = df.drop(columns=[col for col in drop_columns if col in df.columns])

            # Drop duplicates and remove invalid rows
            df = df.drop_duplicates()
            df = df[
                df["customer_id"].notna() & (df["customer_id"] != "Unknown") &
                df["country"].notna() & (df["country"] != "Unknown") &
                df["date_of_birth"].notna() & df["registration_date"].notna()
            ]

            # Fix numeric outliers and missing values
            for col in ["avg_order_value", "total_spent", "loyalty_score", "account_age_days"]:
                if col in df.columns:
                    df[col] = df[col].abs()
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan

            # Fill missing numeric values with random values within range
            rng = np.random.default_rng(seed=42)
            for col in numeric_columns:
                if col in df.columns:
                    valid_values = df[col].dropna()
                    if not valid_values.empty:
                        nan_mask = df[col].isna()
                        df.loc[nan_mask, col] = rng.uniform(
                            low=valid_values.min(), 
                            high=valid_values.max(), 
                            size=nan_mask.sum()
                        )

            # Categorical columns - mode imputation
            categorical_cols = ["customer_status", "newsletter_subscribed", "preferred_language",
                              "currency_preference", "marketing_consent", "gender", "credit_tier"]
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].replace({"Unknown": np.nan, "unknown": np.nan, "": np.nan, pd.NA: np.nan})
                    if not df[col].dropna().empty:
                        mode_value = df[col].mode(dropna=True)[0]
                        df[col] = df[col].fillna(mode_value)

            print("Customers cleaning completed")
            cleaned_tables["customers"] = df

        # ----------------------------- ORDERS TABLE -----------------------------
        if "orders" in dataframes:
            print("Cleaning orders table")
            df1 = dataframes["orders"].copy()

            # ID columns
            id_columns = ["order_id", "customer_id", "product_id"]
            for col in id_columns:
                if col in df1.columns:
                    df1[col] = df1[col].astype(str).str.strip().replace(["", "nan", "None", "NULL"], np.nan)

            # Date columns
            date_columns = ["order_date", "estimated_delivery", "actual_delivery"]
            for col in date_columns:
                if col in df1.columns:
                    df1[col] = pd.to_datetime(df1[col], errors="coerce", infer_datetime_format=True)

            df1["order_date_date"] = df1["order_date"].dt.date
            df1["order_date_time"] = df1["order_date"].dt.time
            df1.drop(columns=["order_date"], inplace=True, errors="ignore")

            # Numeric columns
            numeric_columns = ["order_amount", "quantity", "shipping_cost", "tax_amount", "total_amount", "discount_amount"]
            for col in numeric_columns:
                if col in df1.columns:
                    df1[col] = df1[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
                    df1[col] = pd.to_numeric(df1[col], errors="coerce")
                    df1[col] = df1[col].abs()
                    if col == "quantity":
                        df1[col] = df1[col].round(0)
                    else:
                        df1[col] = df1[col].round(3)

            # Helper to normalize missing values
            def _normalize_missing(series):
                return series.replace(
                    to_replace=["", "none", "nan", "null", "n/a", "na", "None", "NONE", "NULL", "Unknown", "UNKNOWN"],
                    value=np.nan
                )

            # Categorical columns
            if "payment_method" in df1.columns:
                df1["payment_method"] = (
                    df1["payment_method"].astype(str).str.strip().str.upper()
                    .replace({
                        "CREDIT CARD": "CREDIT CARD", "CREDIT_CARD": "CREDIT CARD", "CARD": "CREDIT CARD",
                        "PAYPAL": "PAYPAL", "APPLE PAY": "APPLE PAY", "APPLE_PAY": "APPLE PAY",
                        "GOOGLE PAY": "GOOGLE PAY", "GOOGLE_PAY": "GOOGLE PAY",
                        "BANK TRANSFER": "BANK TRANSFER", "BANK_TRANSFER": "BANK TRANSFER", "CASH": "CASH"
                    })
                    .pipe(_normalize_missing)
                    .str.title()
                )
                df1["payment_method"] = df1["payment_method"].fillna(df1["payment_method"].mode(dropna=True)[0])

            if "shipping_method" in df1.columns:
                df1["shipping_method"] = df1["shipping_method"].astype(str).str.strip().str.title().pipe(_normalize_missing)
                df1["shipping_method"] = df1["shipping_method"].fillna(df1["shipping_method"].mode(dropna=True)[0])

            if "order_status" in df1.columns:
                df1["order_status"] = df1["order_status"].astype(str).str.strip().str.title().pipe(_normalize_missing)
                df1["order_status"] = df1["order_status"].fillna(df1["order_status"].mode(dropna=True)[0])

            if "payment_status" in df1.columns:
                df1["payment_status"] = df1["payment_status"].astype(str).str.strip().str.lower()

            # Currency
            if "currency" in df1.columns:
                currency_map = {"USD": "USD", "EUR": "EUR", "GBP": "GBP"}
                df1["currency"] = (
                    df1["currency"].astype(str).str.strip().str.upper()
                    .replace(currency_map)
                    .pipe(_normalize_missing)
                )
                df1["currency"] = df1["currency"].fillna(df1["currency"].mode(dropna=True)[0])

            # Warehouse ID
            if "warehouse_id" in df1.columns:
                df1["warehouse_id"] = df1["warehouse_id"].astype(str).str.strip().pipe(_normalize_missing)
                df1["warehouse_id"] = df1["warehouse_id"].fillna(df1["warehouse_id"].mode(dropna=True)[0])

            # Channel
            if "channel" in df1.columns:
                df1["channel"] = df1["channel"].astype(str).str.strip().pipe(_normalize_missing).str.title()
                df1["channel"] = df1["channel"].fillna(df1["channel"].mode(dropna=True)[0])

            # Convert to USD
            try:
                rates = get_exchange_rates("USD")
                order_money_columns = ["order_amount", "shipping_cost", "tax_amount", "total_amount", "discount_amount"]
                for col in order_money_columns:
                    if col in df1.columns:
                        df1[f"{col}_usd"] = df1.apply(
                            lambda row: row[col] / rates.get(row["currency"], 1.0) 
                            if pd.notna(row[col]) and row["currency"] in rates else np.nan,
                            axis=1
                        )
            except:
                # Fallback if exchange rate API fails
                for col in order_money_columns:
                    if col in df1.columns:
                        df1[f"{col}_usd"] = df1[col]

            # Drop unwanted columns
            drop_columns = ["shipping_address", "sales_rep_id", "return_reason", "discount_code", "notes", "shipping_address_same"]
            df1 = df1.drop(columns=[col for col in drop_columns if col in df1.columns])

            cleaned_tables["orders"] = df1
            print("Orders cleaning completed")

        # ----------------------------- REVIEWS TABLE -----------------------------
        if "reviews" in dataframes:
            print("Cleaning reviews table")
            df2 = dataframes["reviews"].copy()

            # ID columns
            id_columns = ["review_id", "customer_id", "product_id", "order_id"]
            for col in id_columns:
                if col in df2.columns:
                    df2[col] = df2[col].astype(str).str.strip().replace(["", pd.NA], np.nan)

            # Date columns
            if "review_date" in df2.columns:
                df2["review_date"] = pd.to_datetime(df2["review_date"], errors="coerce")

            # Numeric columns
            numeric_cols = ["rating", "value_for_money", "helpful", "unhelpful"]
            for col in numeric_cols:
                if col in df2.columns:
                    df2[col] = pd.to_numeric(df2[col], errors="coerce")
                    df2[col] = df2[col].abs()
                    q1 = df2[col].quantile(0.25)
                    q3 = df2[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    df2.loc[(df2[col] < lower) | (df2[col] > upper), col] = np.nan

            # Categorical columns
            def _normalize_missing(s):
                return s.replace(to_replace=["", "none", "nan", "null", "n/a", "na"], value=np.nan)

            if "language" in df2.columns:
                df2["language"] = (
                    df2["language"].astype(str).str.strip().str.upper()
                    .replace({"ENGLISH": "EN", "NONE": np.nan, "NAN": np.nan})
                    .pipe(_normalize_missing)
                )
                df2["language"] = df2["language"].fillna(df2["language"].mode(dropna=True)[0])

            # Boolean columns
            boolean_columns = ["verified_purchase", "would_recommend"]
            boolean_map = {
                "1": True, "Y": True, "YES": True, "TRUE": True,
                "0": False, "N": False, "NO": False, "FALSE": False
            }
            for col in boolean_columns:
                if col in df2.columns:
                    df2[col] = (
                        df2[col].astype(str).str.strip().str.upper()
                        .replace(boolean_map)
                        .replace(["", "NaN", "None", "NONE"], np.nan)
                    )
                    if not df2[col].dropna().empty:
                        df2[col] = df2[col].fillna(df2[col].mode(dropna=True)[0])

            if "review_status" in df2.columns:
                df2["review_status"] = df2["review_status"].astype(str).str.strip().pipe(_normalize_missing).str.title()
                df2["review_status"] = df2["review_status"].fillna(df2["review_status"].mode(dropna=True)[0])

            if "product_condition" in df2.columns:
                df2["product_condition"] = df2["product_condition"].str.strip().pipe(_normalize_missing)
                if not df2["product_condition"].dropna().empty:
                    df2["product_condition"] = df2["product_condition"].fillna(df2["product_condition"].mode(dropna=True)[0])
                df2["product_condition"] = df2["product_condition"].str.title()

            # Drop unwanted columns
            columns_to_drop = ["response_from_seller", "response_date", "delivery_rating", 
                             "reviewer_expertise", "review_text", "review_title"]
            df2 = df2.drop(columns=[col for col in columns_to_drop if col in df2.columns])

            print("Reviews cleaning completed")
            cleaned_tables["reviews"] = df2

        # ----------------------------- PRODUCTS TABLE -----------------------------
        if "products" in dataframes:
            print("Cleaning products table")
            df3 = dataframes["products"].copy()

            # Product ID
            df3["product_id"] = (
                df3["product_id"].astype(str).str.strip()
                .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                .fillna("Unknown")
            )

            # SKU
            df3["sku"] = df3["sku"].astype(str).str.strip()
            df3["sku"] = df3["sku"].replace(["", "nan", "none", "None", "NONE", "null"], np.nan)

            # Generate missing SKUs
            def generate_sku():
                num_part = random.randint(10000, 99999)
                letter_part = ''.join(random.choices(string.ascii_uppercase, k=3))
                return f"SKU-{num_part}-{letter_part}"

            df3["sku"] = df3["sku"].apply(lambda x: generate_sku() if pd.isna(x) else x)
            df3["sku"] = df3["sku"].astype(str)

            # Categories
            for col in ["main_category", "sub_category"]:
                if col in df3.columns:
                    df3[col] = (
                        df3[col].astype(str).str.strip().str.title()
                        .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                    )

            # Brand
            if "brand" in df3.columns:
                df3["brand"] = (
                    df3["brand"].astype(str).str.strip().str.title()
                    .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                    .fillna("Unknown")
                )

            # Tax category
            if "tax_category" in df3.columns:
                df3["tax_category"] = (
                    df3["tax_category"].astype(str).str.strip().str.title()
                    .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                )

            # Product status
            if "product_status" in df3.columns:
                df3["product_status"] = (
                    df3["product_status"].astype(str).str.strip().str.title()
                    .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                )

            # Boolean columns
            bool_map = {
                "true": True, "false": False, "yes": True, "no": False,
                "Y": True, "N": False, "y": True, "n": False,
                "1": True, "0": False, True: True, False: False
            }
            for col in ["is_digital", "requires_shipping"]:
                if col in df3.columns:
                    df3[col] = (
                        df3[col].astype(str).str.strip().str.title()
                        .replace(bool_map)
                        .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                    )

            # Currency
            currency_map = {"$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR",
                           "gbp": "GBP", "£": "GBP", "¥": "JPY", "jpy": "JPY"}
            if "currency" in df3.columns:
                df3["currency"] = (
                    df3["currency"].astype(str).str.strip().str.upper()
                    .replace(currency_map)
                    .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                )

            # Numeric columns
            for col in ["stock_quantity", "weight_kg", "review_count", "warranty_months"]:
                if col in df3.columns:
                    df3[col] = pd.to_numeric(df3[col], errors="coerce")

            # Convert negatives to absolute values
            for col in ["price", "cost", "stock_quantity", "warranty_months"]:
                if col in df3.columns:
                    df3[col] = df3[col].abs()

            # Drop unwanted columns
            drop_cols = ["product_name", "sub_category", "dimensions", "color",
                        "size", "material", "supplier_id", "description", "tags", "manufacturer"]
            df3 = df3.drop(columns=[c for c in drop_cols if c in df3.columns])

            # Outlier clipping
            num_cols = ["price", "cost", "stock_quantity"]
            def clip_upper_iqr(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                upper = Q3 + 1.5 * IQR
                return series.apply(lambda x: upper if (pd.notna(x) and x > upper) else x)

            for col in num_cols:
                if col in df3.columns:
                    df3[col] = clip_upper_iqr(df3[col])

            # Fill missing values
            def fill_random_within_range(series):
                non_null = series.dropna()
                if non_null.empty:
                    return series.fillna(0)
                low, high = non_null.min(), non_null.max()
                return series.apply(lambda x: np.random.uniform(low, high) if pd.isna(x) else x)

            for col in ["stock_quantity", "weight_kg", "review_count"]:
                if col in df3.columns:
                    df3[col] = fill_random_within_range(df3[col])

            # Generate cost/price if missing
            df3["cost"] = pd.to_numeric(df3["cost"], errors="coerce")
            df3["price"] = pd.to_numeric(df3["price"], errors="coerce")

            cost_min, cost_max = df3["cost"].min(), df3["cost"].max()
            if pd.isna(cost_min):
                cost_min, cost_max = 10, 1000

            def generate_cost(row):
                if pd.isna(row["cost"]) and not pd.isna(row["price"]):
                    return float(row["price"] / np.random.uniform(1.8, 2.2))
                if pd.isna(row["cost"]) and pd.isna(row["price"]):
                    return float(np.random.uniform(cost_min, cost_max))
                return float(row["cost"])

            df3["cost"] = df3.apply(generate_cost, axis=1)
            df3["price"] = df3.apply(
                lambda row: float(row["cost"] * np.random.uniform(1.8, 2.2))
                if pd.isna(row["price"]) else float(row["price"]),
                axis=1
            )

            df3["cost"] = df3["cost"].round(2)
            df3["price"] = df3["price"].round(2)

            # Warranty months
            if "warranty_months" in df3.columns:
                df3["warranty_months"] = df3["warranty_months"].apply(
                    lambda x: np.random.choice([12, 24, 36, 48, 60]) if pd.isna(x) else x
                )

            # Country of origin
            if "country_of_origin" in df3.columns:
                def standardize_country(val):
                    if pd.isna(val):
                        return np.nan
                    try:
                        return pycountry.countries.lookup(val).name
                    except:
                        return val

                df3["country_of_origin"] = df3["country_of_origin"].astype(str).str.strip()
                df3["country_of_origin"] = df3["country_of_origin"].replace(
                    ["", "nan", "none", "None", "NONE", "null"], np.nan
                )
                df3["country_of_origin"] = df3["country_of_origin"].apply(standardize_country)

            # Mode imputation
            mode_cols = ["main_category", "currency", "product_status", "country_of_origin", 
                        "is_digital", "requires_shipping", "tax_category"]
            for col in mode_cols:
                if col in df3.columns:
                    mode_val = df3[col].mode(dropna=True)[0] if df3[col].notna().any() else "Unknown"
                    df3[col] = df3[col].fillna(mode_val)

            # Rating
            if "rating" in df3.columns:
                df3["rating"] = pd.to_numeric(df3["rating"], errors="coerce")
                df3["rating"] = df3["rating"].apply(
                    lambda x: np.random.randint(1, 6) if pd.isna(x) else int(x)
                )

            print("Products cleaning completed")
            cleaned_tables["products"] = df3

        # Save cleaned tables back to SQLite if requested
        if save_to_sqlite:
            print("\nSaving cleaned tables back to database...")
            for table_name, df_clean in cleaned_tables.items():
                df_clean.to_sql(table_name + "_clean", conn, if_exists="replace", index=False)
                print(f"Saved {table_name}_clean with shape {df_clean.shape}")

        print("\nPipeline completed successfully!")
        return cleaned_tables

    finally:
        conn.close()

def get_exchange_rates(base="USD"):
    """Fetch current exchange rates from API"""
    try:
        url = f"https://open.er-api.com/v6/latest/{base}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()["rates"]
    except:
        # Fallback rates if API fails
        return {"USD": 1.0, "EUR": 0.92, "GBP": 0.79}

# =========================================================================
# ANALYTICS FUNCTIONS
# =========================================================================

@st.cache_data
def monthly_active_customers_analytics(cleaned_data):
    """Monthly active customers over time"""
    df2 = cleaned_data["orders"]
    monthly_active = (
        df2.dropna(subset=["order_date_date"])
        .assign(year_month=lambda x: pd.to_datetime(x["order_date_date"]).dt.to_period("M").astype(str))
        .groupby("year_month", as_index=False)
        .agg(active_customers=("customer_id", "nunique"))
        .sort_values("year_month")
    )

    fig = px.line(
        monthly_active,
        x="year_month",
        y="active_customers",
        title="Monthly Active Customers Over Time",
        labels={"year_month": "Year-Month", "active_customers": "Active Customers"},
        markers=True
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
    return monthly_active, fig

@st.cache_data
def churn_customer_attributes(cleaned_data, sample_size=100):
    """Analyze churn by customer attributes"""
    df_customers = cleaned_data["customers"]
    df_orders = cleaned_data["orders"]

    monthly_customers = (
        df_orders[["customer_id", "order_date_date"]]
        .dropna()
        .assign(year_month=lambda x: pd.to_datetime(x["order_date_date"]).dt.to_period("M").astype(str))
        .drop_duplicates()
    )

    monthly_customers["next_month"] = (
        pd.to_datetime(monthly_customers["year_month"] + "-01") + pd.offsets.MonthBegin(1)
    ).dt.to_period("M").astype(str)

    churn_flag = monthly_customers.merge(
        monthly_customers[["customer_id", "year_month"]],
        left_on=["customer_id", "next_month"],
        right_on=["customer_id", "year_month"],
        how="left",
        indicator=True
    )
    churn_flag["churned"] = (churn_flag["_merge"] == "left_only").astype(int)
    churn_flag = churn_flag.rename(columns={"year_month_x": "year_month"})[["customer_id", "year_month", "churned"]]

    customers_attr = df_customers.copy()
    customers_attr["date_of_birth"] = pd.to_datetime(customers_attr["date_of_birth"], errors="coerce")
    customers_attr["age"] = ((pd.Timestamp("now") - customers_attr["date_of_birth"]).dt.days / 365.25).round(0)

    churn_attributes = churn_flag.merge(
        customers_attr[["customer_id", "country", "credit_tier", "loyalty_score", "age", 
                       "newsletter_subscribed", "marketing_consent"]],
        on="customer_id",
        how="left"
    ).dropna(subset=["age"])

    return churn_attributes.head(sample_size)

@st.cache_data
def churn_customer_churn_fig(cleaned_data, sample_size=100):
    """Create churn scatter plot"""
    churn_attributes = churn_customer_attributes(cleaned_data, sample_size)
    fig = px.scatter(
        churn_attributes,
        x="loyalty_score",
        y="age",
        size="churned",
        color="credit_tier",
        symbol="newsletter_subscribed",
        hover_name="customer_id",
        title="Customer Churn by Loyalty Score and Age",
        size_max=20
    )
    return fig

@st.cache_data
def monthly_churn_rate(df2):
    """Calculate monthly churn rate"""
    orders = (
        df2.dropna(subset=["order_date_date", "customer_id"])
        .assign(year_month=lambda x: pd.to_datetime(x["order_date_date"]).dt.to_period("M").astype(str))
        [["customer_id", "year_month"]].drop_duplicates()
    )

    next_month = orders.copy()
    next_month["year_month"] = (
        pd.to_datetime(next_month["year_month"] + "-01") + pd.DateOffset(months=1)
    ).dt.to_period("M").astype(str)

    merged = orders.merge(next_month, on=["customer_id", "year_month"], how="left", indicator=True)
    churn_df = (
        merged.groupby("year_month", as_index=False)
        .agg(
            active_customers=("customer_id", "nunique"),
            churned_customers=("_merge", lambda x: (x == "left_only").sum())
        )
    )
    churn_df["churn_rate"] = (churn_df["churned_customers"] / churn_df["active_customers"]).round(4)
    return churn_df.sort_values("year_month")

@st.cache_data
def plot_monthly_churn(churn_df):
    """Plot monthly churn"""
    fig = px.bar(
        churn_df,
        x="year_month",
        y=["active_customers", "churned_customers"],
        barmode="group",
        title="Active vs Churned Customers per Month",
        labels={"value": "Number of Customers", "year_month": "Month", "variable": "Customer Type"},
        color_discrete_map={"active_customers": "green", "churned_customers": "red"},
        height=500
    )
    fig.update_yaxes(range=[0, churn_df["active_customers"].max() * 1.1])
    return fig

@st.cache_data
def customer_churn_flag(df_orders, churn_threshold_months=3):
    """Create churn flag for customers"""
    df = df_orders.copy()
    df["order_date_date"] = pd.to_datetime(df["order_date_date"], errors="coerce")
    df = df.dropna(subset=["order_date_date"])

    df["year_month"] = df["order_date_date"].dt.to_period("M").astype(str)

    last_purchase = (
        df.groupby("customer_id")["order_date_date"]
        .max().reset_index(name="last_order_date")
    )

    reference_date = df["order_date_date"].max()
    last_purchase["months_inactive"] = (
        (reference_date - last_purchase["last_order_date"]) / pd.Timedelta(days=30)
    )

    last_purchase["churned"] = (last_purchase["months_inactive"] >= churn_threshold_months).astype(int)

    churn_df = (
        df[["customer_id", "year_month"]].drop_duplicates()
        .merge(last_purchase[["customer_id", "churned"]], on="customer_id", how="left")
    )
    return churn_df

@st.cache_data
def plot_customer_churn_scatter(churn_flag):
    """Plot customer churn timeline"""
    fig_scatter = px.scatter(
        churn_flag.assign(churned_str=churn_flag["churned"].astype(str)),
        x="customer_id",
        y="year_month",
        color="churned_str",
        title="Customer Churn Over Time",
        labels={"customer_id": "Customer ID", "year_month": "Year-Month", "churned_str": "Churned"},
        color_discrete_map={"0": "green", "1": "red"}
    )
    fig_scatter.update_yaxes(categoryorder="category ascending")
    return fig_scatter

@st.cache_data
def revenue_by_country_analytics(cleaned_data, top_n=50):
    """Revenue by country"""
    df_customers = cleaned_data["customers"]
    revenue_by_country = (
        df_customers.groupby("country", as_index=False)
        .agg(customer_count=("customer_id", "nunique"), revenue=("total_spent", "sum"))
        .sort_values("revenue", ascending=False)
        .head(top_n)
    )
    return revenue_by_country

@st.cache_data
def revenue_by_country_map(cleaned_data, top_n=None):
    """World map of revenue by country"""
    revenue_by_country = revenue_by_country_analytics(cleaned_data, top_n)
    if revenue_by_country.empty:
        return None
        
    fig_map = px.choropleth(
        revenue_by_country,
        locations="country",
        locationmode="country names",
        color="revenue",
        hover_name="country",
        color_continuous_scale="Viridis",
        title="Revenue by Country"
    )
    fig_map.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth")
    )
    return fig_map

@st.cache_data
def loyalty_analysis_analytics(cleaned_data, top_n=100):
    """Loyalty analysis for top customers"""
    df_customers = cleaned_data["customers"]
    df_orders = cleaned_data["orders"]

    customers_ranked = df_customers.copy()
    customers_ranked["rank"] = customers_ranked["total_spent"].rank(method="average", pct=True)
    top_customers = customers_ranked[customers_ranked["rank"] >= 0.9]

    merged = df_orders.merge(top_customers, on="customer_id", how="inner")

    loyalty_analysis = (
        merged.groupby(
            ["customer_id", "country", "credit_tier", "loyalty_score", "newsletter_subscribed"],
            as_index=False
        )
        .agg(
            purchase_frequency=("order_id", "count"),
            lifetime_value=("total_amount_usd", "sum")
        )
        .sort_values("lifetime_value", ascending=False)
        .head(top_n)
    )
    return loyalty_analysis

@st.cache_data
def plot_loyalty_analysis(loyalty_analysis):
    """Plot loyalty vs LTV"""
    fig = px.scatter(
        loyalty_analysis,
        x="purchase_frequency",
        y="lifetime_value",
        color="credit_tier",
        size="loyalty_score",
        symbol="newsletter_subscribed",
        hover_data=["customer_id", "country"],
        title="Customer Purchase Frequency vs Lifetime Value"
    )
    fig.update_layout(
        xaxis_title="Purchase Frequency",
        yaxis_title="Lifetime Value (LTV)",
        height=450
    )
    return fig

@st.cache_data
def volume_driver_analysis(cleaned_data, top_n=100):
    """Volume driver analysis by product"""
    df_products = cleaned_data["products"]
    df_orders = cleaned_data["orders"]

    df = df_orders.merge(df_products, on="product_id", how="left")
    volume_drivers = (
        df.groupby(["product_id"], as_index=False)
        .agg(
            revenue=("total_amount_usd", "sum"),
            cost=("quantity", lambda x: (x * df.loc[x.index, "cost"]).sum())
        )
    )

    volume_drivers["profit"] = volume_drivers["revenue"] - volume_drivers["cost"]
    volume_drivers["margin"] = volume_drivers["profit"] / volume_drivers["revenue"]

    return volume_drivers.sort_values("revenue", ascending=False).head(top_n)

@st.cache_data
def plot_volume_drivers(volume_drivers_df, top_n=10):
    """Plot top volume drivers"""
    top_revenue = volume_drivers_df.head(top_n)
    fig = px.bar(
        top_revenue,
        x="revenue",
        y="product_id",
        orientation="h",
        color="margin",
        text="profit",
        title="Top 10 Products by Revenue (Volume Drivers vs Profit Engines)",
        labels={"revenue": "Revenue (USD)", "product_id": "Product", "margin": "Profit Margin"},
        color_continuous_scale="Viridis"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
    return fig

@st.cache_data
def bottom_products_analysis(cleaned_data, bottom_pct=0.05, max_rows=100):
    """Identify bottom-performing products"""
    df_products = cleaned_data["products"]
    df_orders = cleaned_data["orders"]
    df_reviews = cleaned_data["reviews"]

    product_sales = df_orders.groupby("product_id", as_index=False).agg(revenue=("total_amount_usd", "sum"))
    product_ratings = df_reviews.groupby("product_id", as_index=False).agg(avg_rating=("rating", "mean"))
    product_perf = product_sales.merge(product_ratings, on="product_id", how="inner").dropna()
    product_perf = product_perf.sort_values(["revenue", "avg_rating"], ascending=[True, True]).reset_index(drop=True)

    total_products = len(product_perf)
    cutoff = max(1, int(bottom_pct * total_products))
    bottom_products = product_perf.head(min(cutoff, max_rows))

    fig = px.scatter(
        bottom_products,
        x="avg_rating",
        y="revenue",
        size="revenue",
        color="avg_rating",
        hover_name="product_id",
        title="Bottom Performing Products: Revenue vs Average Rating",
        labels={"avg_rating": "Average Rating", "revenue": "Revenue (USD)"},
        color_continuous_scale="RdYlGn_r"
    )
    fig.update_layout(height=450)
    return bottom_products, fig

@st.cache_data
def compute_kbi(cleaned_data):
    """Compute key business indicators"""
    customers = cleaned_data["customers"]
    orders = cleaned_data["orders"]

    active_statuses = ["active", "gold", "silver", "premium"]
    inactive_statuses = ["inactive", "suspended"]

    total_customers = customers["customer_id"].nunique()
    active_customers = customers.loc[
        customers["customer_status"].str.lower().isin(active_statuses), "customer_id"
    ].nunique()
    inactive_customers = customers.loc[
        customers["customer_status"].str.lower().isin(inactive_statuses), "customer_id"
    ].nunique()
    unknown_status_customers = total_customers - (active_customers + inactive_customers)

    total_orders = orders["order_id"].nunique()
    total_revenue = round(orders["total_amount_usd"].sum(), 2)
    avg_order_value = round(orders["total_amount_usd"].mean(), 2) if total_orders > 0 else 0
    avg_orders_per_customer = round(total_orders / total_customers, 2) if total_customers > 0 else 0
    active_customer_pct = round((active_customers / total_customers) * 100, 2) if total_customers > 0 else 0

    return pd.DataFrame({
        "Metric": [
            "Total Customers", "Active Customers", "Inactive Customers",
            "Customers with Unknown Status", "Active Customer Percentage (%)",
            "Total Orders", "Total Revenue (USD)", "Average Order Value (USD)",
            "Average Orders per Customer"
        ],
        "Value": [
            total_customers, active_customers, inactive_customers,
            unknown_status_customers, active_customer_pct, total_orders,
            total_revenue, avg_order_value, avg_orders_per_customer
        ]
    })

@st.cache_data
def delivery_time_rating_analysis(cleaned_data, limit=200):
    """Analyze delivery time vs rating"""
    df_products = cleaned_data["products"]
    df_orders = cleaned_data["orders"]

    delivery_rating_df = (
        df_orders.merge(df_products[["product_id", "rating"]], on="product_id", how="inner")
        .assign(
            delivery_time_days=lambda x: (
                pd.to_datetime(x["actual_delivery"], errors="coerce") -
                pd.to_datetime(x["order_date_date"], errors="coerce")
            ).dt.days
        )
        [["product_id", "rating", "delivery_time_days"]]
        .dropna()
        .head(limit)
    )
    return delivery_rating_df

@st.cache_data
def plot_delivery_time_vs_rating(df):
    """Plot delivery time vs rating"""
    fig = px.scatter(
        df,
        x="delivery_time_days",
        y="rating",
        trendline="ols",
        title="Relationship Between Delivery Time and Product Rating",
        labels={"delivery_time_days": "Delivery Time (Days)", "rating": "Product Rating"}
    )
    return fig

@st.cache_data
def delivery_rate_by_warehouse(cleaned_data):
    """Calculate delivery rate by warehouse"""
    df_orders = cleaned_data["orders"].copy()
    analytics_df = (
        df_orders.groupby("warehouse_id", as_index=False)
        .agg(
            shipped_orders=("actual_delivery", lambda x: x.notna().sum()),
            total_orders=("order_id", "nunique")
        )
    )
    analytics_df["delivery_rate"] = (
        analytics_df["shipped_orders"] / analytics_df["total_orders"] * 100
    ).round(2)
    return analytics_df

@st.cache_data
def plot_delivery_rate_by_warehouse(analytics_df):
    """Plot delivery rate by warehouse"""
    fig = px.bar(
        analytics_df,
        x="warehouse_id",
        y="delivery_rate",
        title="Delivery Rate by Warehouse",
        labels={"warehouse_id": "Warehouse ID", "delivery_rate": "Delivery Rate (%)"},
        text="delivery_rate"
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis_ticksuffix="%", xaxis_tickangle=-30, height=450)
    return fig

@st.cache_data
def payment_failure_rate_analysis(cleaned_data):
    """Calculate payment failure rate by method"""
    orders_df = cleaned_data["orders"].copy()
    if "payment_status" not in orders_df.columns:
        return pd.DataFrame(columns=["payment_method", "all_payments", "failed_payments", "failed_payment_rate"])
        
    orders_df["payment_status"] = orders_df["payment_status"].astype(str).str.strip().str.lower()
    payment_failure_df = (
        orders_df.groupby("payment_method", as_index=False)
        .agg(
            all_payments=("order_id", "count"),
            failed_payments=("payment_status", lambda x: (x == "failed").sum())
        )
    )
    payment_failure_df["failed_payment_rate"] = (
        100 * payment_failure_df["failed_payments"] / payment_failure_df["all_payments"].replace(0, np.nan)
    ).round(2)
    return payment_failure_df.dropna(subset=["failed_payment_rate"]).sort_values("failed_payment_rate", ascending=False)

@st.cache_data
def plot_payment_failure_rate(payment_df):
    """Plot payment failure rate"""
    if payment_df.empty:
        return None
        
    fig = px.bar(
        payment_df,
        x="payment_method",
        y="failed_payment_rate",
        title="Failed Payment Rate by Payment Method",
        labels={"payment_method": "Payment Method", "failed_payment_rate": "Failed Payment Rate (%)"},
        text="failed_payment_rate"
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        yaxis_ticksuffix="%",
        xaxis_tickangle=-30,
        height=450,
        yaxis_range=[0, payment_df["failed_payment_rate"].max() * 1.2]
    )
    return fig

@st.cache_data
def compute_yearly_orders_profit(cleaned_data):
    """Compute yearly orders and profit"""
    df2 = cleaned_data["orders"].copy()
    df1 = cleaned_data["products"].copy()

    df2["order_date_date"] = pd.to_datetime(df2["order_date_date"], errors="coerce")
    orders_products = df2.merge(df1[["product_id", "price", "cost"]], on="product_id", how="inner")
    
    orders_products["profit_usd"] = (
        orders_products["total_amount_usd"] - (orders_products["cost"] * orders_products["quantity"])
    )

    yearly_orders_profit = (
        orders_products.dropna(subset=["order_date_date"])
        .assign(order_year=lambda x: x["order_date_date"].dt.year)
        .groupby("order_year", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_profit_usd=("profit_usd", "sum")
        )
        .sort_values("order_year")
    )
    return yearly_orders_profit

@st.cache_data
def yearly_monthly_sales_profit_analysis(cleaned_data):
    """Yearly and monthly sales/profit analysis"""
    df_orders = cleaned_data["orders"]
    df_products = cleaned_data["products"]

    df = df_orders.merge(df_products[["product_id", "price", "cost"]], on="product_id", how="inner")
    df = df.dropna(subset=["order_date_date"])
    df["order_date_date"] = pd.to_datetime(df["order_date_date"])
    df["order_year"] = df["order_date_date"].dt.year.astype(str)
    df["order_month"] = df["order_date_date"].dt.month.astype(str).str.zfill(2)

    # Currency conversion for profit
    def profit_usd(row):
        profit = row["price"] - row["cost"]
        if row["currency"] == "USD":
            return profit
        elif row["currency"] == "EUR":
            return profit * 1.09
        elif row["currency"] == "GBP":
            return profit * 1.27
        return 0

    df["profit_usd"] = df.apply(profit_usd, axis=1)

    summary = (
        df.groupby(["order_year", "order_month"], as_index=False)
        .agg(
            sales_usd=("total_amount_usd", "sum"),
            total_orders=("order_id", "count"),
            profit_usd=("profit_usd", "sum")
        )
        .round(2)
        .sort_values(["order_year", "order_month"])
    )
    return summary

@st.cache_data
def plot_monthly_sales_profit_trend(df_year_month):
    """Interactive monthly sales & profit trend"""
    fig = go.Figure()
    years = sorted(df_year_month["order_year"].unique())
    n_years = len(years)

    for idx, year in enumerate(years):
        df_year = df_year_month[df_year_month["order_year"] == year]

        fig.add_trace(
            go.Scatter(
                x=df_year["order_month"],
                y=df_year["sales_usd"],
                mode="lines+markers",
                name=f"Sales – {year}",
                yaxis="y",
                visible=True if idx == 0 else False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_year["order_month"],
                y=df_year["profit_usd"],
                mode="lines+markers",
                name=f"Profit – {year}",
                yaxis="y2",
                visible=True if idx == 0 else False
            )
        )

    buttons = []
    for idx, year in enumerate(years):
        visible = [False] * (n_years * 2)
        visible[idx * 2] = True
        visible[idx * 2 + 1] = True

        buttons.append(
            dict(
                label=str(year),
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Monthly Sales & Profit Trend – {year}"}
                ],
            )
        )

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0.02, y=1.15)],
        title=f"Monthly Sales & Profit Trend – {years[0]}",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Sales (USD)", tickprefix="$"),
        yaxis2=dict(title="Profit (USD)", overlaying="y", side="right", tickprefix="$"),
        height=450,
        legend_title="Metric",
        template="plotly_white"
    )
    return fig

@st.cache_data
def plot_monthly_orders_trend(df_year_month):
    """Interactive monthly orders trend"""
    fig = go.Figure()
    years = sorted(df_year_month["order_year"].unique())

    for i, year in enumerate(years):
        df_year = df_year_month[df_year_month["order_year"] == year]

        fig.add_trace(
            go.Scatter(
                x=df_year["order_month"],
                y=df_year["total_orders"],
                mode="lines+markers",
                name=str(year),
                visible=True if i == 0 else False
            )
        )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": str(year),
                        "method": "update",
                        "args": [
                            {"visible": [y == year for y in years]},
                            {"title": f"Monthly Orders Trend – {year}"}
                        ],
                    }
                    for year in years
                ],
                "direction": "down",
                "x": 0.02,
                "y": 1.15,
            }
        ],
        title=f"Monthly Orders Trend – {years[0]}",
        xaxis_title="Month",
        yaxis_title="Total Orders",
        height=450
    )
    return fig

@st.cache_data
def currency_profit_sales_analysis(cleaned_data):
    """Profit and sales by currency"""
    df_orders = cleaned_data["orders"]
    df_products = cleaned_data["products"]

    df = df_orders.merge(df_products[["product_id", "price", "cost"]], on="product_id", how="inner")
    df["profit_usd"] = np.where(
        df["currency"] == "USD", (df["price"] - df["cost"]),
        np.where(
            df["currency"] == "EUR", (df["price"] - df["cost"]) * 1.09,
            np.where(df["currency"] == "GBP", (df["price"] - df["cost"]) * 1.27, 0)
        )
    )

    currency_df = (
        df.groupby("currency", as_index=False)
        .agg(
            total_sales_usd=("total_amount_usd", "sum"),
            total_profit_usd=("profit_usd", "sum")
        )
        .round(2)
        .sort_values(["total_profit_usd", "total_sales_usd"], ascending=False)
    )
    return currency_df

@st.cache_data
def plot_profit_by_currency(currency_df):
    """Donut chart of profit by currency"""
    fig = px.pie(
        currency_df,
        names="currency",
        values="total_profit_usd",
        title="Profit Contribution by Currency",
        hole=0.4
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Profit: $%{value:,.2f}<extra></extra>"
    )
    fig.update_layout(height=450)
    return fig

@st.cache_data
def customer_spending_tier_analysis(cleaned_data):
    """Customer distribution by spending tier"""
    df_customers = cleaned_data["customers"].copy()
    df_customers = df_customers.dropna(subset=["total_spent"])

    df_customers["spending_tier"] = pd.cut(
        df_customers["total_spent"],
        bins=[-np.inf, 100, 1000, 5000, 10000, np.inf],
        labels=[
            "Low (<100)",
            "Occasional (100-1k)",
            "Regular (1k-5k)",
            "Premium (5k-10k)",
            "VIP (>10k)"
        ]
    )

    spending_summary = (
        df_customers.groupby("spending_tier", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            avg_spent=("total_spent", "mean"),
            avg_account_age=("account_age_days", "mean")
        )
        .round({"avg_spent": 2, "avg_account_age": 1})
        .sort_values("avg_spent", ascending=False)
    )
    return spending_summary

@st.cache_data
def plot_customer_spending_tiers(spending_df):
    """Pie chart of customer spending tiers"""
    fig = px.pie(
        spending_df,
        names="spending_tier",
        values="customer_count",
        hover_data=["avg_spent", "avg_account_age"],
        title="Customer Distribution by Spending Tier"
    )
    fig.update_traces(
        textinfo="percent+label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Customers: %{value}<br>"
            "Avg Spent: $%{customdata[0]:,.2f}<br>"
            "Avg Account Age: %{customdata[1]} days"
            "<extra></extra>"
        )
    )
    fig.update_layout(height=450)
    return fig

# =========================================================================
# MACHINE LEARNING MODELS
# =========================================================================

@st.cache_resource
def train_churn_and_ltv_models(cleaned_data, reference_date="2025-12-30"):
    """Train churn and LTV models"""
    np.random.seed(42)
    
    # Prepare data
    orders = cleaned_data["orders"].copy()
    customers = cleaned_data["customers"].copy()
    
    # Filter orders
    if "order_status" in orders.columns:
        orders = orders[~orders["order_status"].isin(["cancelled", "returned", "failed"])]
    
    orders["order_date_date"] = pd.to_datetime(orders["order_date_date"], errors="coerce")
    orders["total_amount"] = pd.to_numeric(orders["total_amount"], errors="coerce").fillna(0)
    
    # Convert to USD
    currency_rates = {"USD": 1.0, "EUR": 1.1, "GBP": 1.25}
    orders["total_amount_usd"] = orders.apply(
        lambda x: x["total_amount"] * currency_rates.get(x.get("currency", "USD"), 1.0),
        axis=1
    )
    
    orders["customer_id"] = orders["customer_id"].astype(str)
    customers["customer_id"] = customers["customer_id"].astype(str)
    today = pd.Timestamp(reference_date)

    # ======================= CHURN MODEL =======================
    customer_orders = (
        orders.groupby("customer_id")
        .agg(
            last_order_date=("order_date_date", "max"),
            total_orders=("order_id", "count"),
            avg_order_value=("total_amount_usd", "mean")
        )
        .reset_index()
    )

    df_churn = customers.merge(customer_orders, on="customer_id", how="left")

    # SAFE COLUMN RESOLUTION - FIX FOR KEYERROR
    # Check if avg_order_value exists in different forms after merge
    if "avg_order_value" not in df_churn.columns:
        if "avg_order_value_y" in df_churn.columns:
            df_churn["avg_order_value"] = df_churn["avg_order_value_y"]
        elif "avg_order_value_x" in df_churn.columns:
            df_churn["avg_order_value"] = df_churn["avg_order_value_x"]
        else:
            df_churn["avg_order_value"] = 0  # Default if column doesn't exist

    # Same for total_orders
    if "total_orders" not in df_churn.columns:
        if "total_orders_y" in df_churn.columns:
            df_churn["total_orders"] = df_churn["total_orders_y"]
        elif "total_orders_x" in df_churn.columns:
            df_churn["total_orders"] = df_churn["total_orders_x"]
        else:
            df_churn["total_orders"] = 0

    # Fill missing values
    df_churn["total_orders"] = df_churn["total_orders"].fillna(0)
    df_churn["avg_order_value"] = df_churn["avg_order_value"].fillna(0)
    df_churn["last_order_date"] = pd.to_datetime(df_churn["last_order_date"], errors="coerce").fillna(today)
    df_churn["days_since_last_order"] = (today - df_churn["last_order_date"]).dt.days

    # RFM scoring
    df_churn["recency_score"] = df_churn["days_since_last_order"].apply(
        lambda d: 1.0 if d > 180 else 0.7 if d > 90 else 0.4 if d > 30 else 0.1
    )
    df_churn["frequency_score"] = df_churn["total_orders"].apply(
        lambda o: 0.8 if o == 1 else 0.3 if 2 <= o <= 5 else 0.1
    )
    df_churn["monetary_score"] = df_churn["avg_order_value"].apply(
        lambda a: 0.6 if a < 50 else 0.3 if a < 100 else 0.1
    )

    df_churn["churn_risk_score"] = (
        df_churn["recency_score"] * 0.5 +
        df_churn["frequency_score"] * 0.3 +
        df_churn["monetary_score"] * 0.2
    ).round(2)

    df_churn["churn_risk_category"] = pd.cut(
        df_churn["churn_risk_score"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True
    )

    df_churn["target"] = (df_churn["churn_risk_category"] == "High Risk").astype(int)

    # Features
    churn_features = ["total_orders", "avg_order_value", "days_since_last_order", "loyalty_score", "account_age_days"]
    churn_features = [c for c in churn_features if c in df_churn.columns]

    X_churn = df_churn[churn_features].fillna(0)
    y_churn = df_churn["target"]

    if len(y_churn.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
        )
        churn_model = RandomForestClassifier(n_estimators=200, random_state=42)
        churn_model.fit(X_train, y_train)
        churn_preds = churn_model.predict(X_test)
        churn_metrics = classification_report(y_test, churn_preds, output_dict=True)
    else:
        churn_model = None
        churn_metrics = {"1": {"precision": 0, "recall": 0, "f1-score": 0}}

    df_churn["predicted_churn_prob"] = churn_model.predict_proba(X_churn)[:, 1] if churn_model is not None else 0

    # ======================= LTV MODEL =======================
    customer_orders_ltv = (
        orders.groupby("customer_id")
        .agg(
            total_spent_usd=("total_amount_usd", "sum"),
            avg_order_value=("total_amount_usd", "mean"),
            total_orders=("order_id", "count"),
            last_order_date=("order_date_date", "max"),
            first_order_date=("order_date_date", "min")
        )
        .reset_index()
    )

    df_ltv = customers.merge(customer_orders_ltv, on="customer_id", how="left")

    # SAFE COLUMN RESOLUTION FOR LTV
    # Check for avg_order_value column conflicts
    if "avg_order_value" not in df_ltv.columns:
        if "avg_order_value_y" in df_ltv.columns:
            df_ltv["avg_order_value"] = df_ltv["avg_order_value_y"]
        elif "avg_order_value_x" in df_ltv.columns:
            df_ltv["avg_order_value"] = df_ltv["avg_order_value_x"]
        else:
            df_ltv["avg_order_value"] = 0

    # Same for total_spent_usd
    if "total_spent_usd" not in df_ltv.columns:
        if "total_spent_usd_y" in df_ltv.columns:
            df_ltv["total_spent_usd"] = df_ltv["total_spent_usd_y"]
        elif "total_spent_usd_x" in df_ltv.columns:
            df_ltv["total_spent_usd"] = df_ltv["total_spent_usd_x"]
        else:
            df_ltv["total_spent_usd"] = 0

    # Same for total_orders
    if "total_orders" not in df_ltv.columns:
        if "total_orders_y" in df_ltv.columns:
            df_ltv["total_orders"] = df_ltv["total_orders_y"]
        elif "total_orders_x" in df_ltv.columns:
            df_ltv["total_orders"] = df_ltv["total_orders_x"]
        else:
            df_ltv["total_orders"] = 0

    # Fill missing values
    df_ltv["total_spent_usd"] = df_ltv["total_spent_usd"].fillna(0)
    df_ltv["avg_order_value"] = df_ltv["avg_order_value"].fillna(0)
    df_ltv["total_orders"] = df_ltv["total_orders"].fillna(0)
    
    df_ltv["last_order_date"] = pd.to_datetime(df_ltv["last_order_date"], errors="coerce").fillna(today)
    df_ltv["first_order_date"] = pd.to_datetime(df_ltv["first_order_date"], errors="coerce").fillna(today)

    df_ltv["days_since_last_order"] = (today - df_ltv["last_order_date"]).dt.days
    df_ltv["customer_age_days"] = (today - df_ltv["first_order_date"]).dt.days
    df_ltv["order_frequency_days"] = (
        df_ltv["customer_age_days"] / df_ltv["total_orders"].replace(0, np.nan)
    ).fillna(df_ltv["customer_age_days"])

    # Features
    ltv_features = ["total_orders", "avg_order_value", "days_since_last_order", "customer_age_days",
                   "order_frequency_days", "loyalty_score"]
    ltv_features = [c for c in ltv_features if c in df_ltv.columns]

    X_ltv = df_ltv[ltv_features].fillna(0)
    y_ltv = df_ltv["total_spent_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X_ltv, y_ltv, test_size=0.2, random_state=42)

    ltv_model = RandomForestRegressor(n_estimators=200, random_state=42)
    ltv_model.fit(X_train, y_train)
    y_pred = ltv_model.predict(X_test)

    df_ltv["predicted_ltv_usd"] = ltv_model.predict(X_ltv)

    ltv_metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return {
        "churn_df": df_churn,
        "ltv_df": df_ltv,
        "churn_model": churn_model,
        "ltv_model": ltv_model,
        "churn_metrics": churn_metrics,
        "ltv_metrics": ltv_metrics
    }

# =========================================================================
# FEATURE IMPORTANCE (REMOVE CACHING)
# =========================================================================

def get_feature_importance(churn_model, ltv_model, churn_features, ltv_features):
    """
    Extract feature importance for churn and LTV Random Forest models.
    NOTE: This function is not cached because model objects are not hashable.
    The expensive operation (model training) is already cached separately.
    """
    feature_importance = {}

    # -------------------------------
    # Churn feature importance
    # -------------------------------
    if churn_model is not None and hasattr(churn_model, "feature_importances_"):
        churn_importance = pd.DataFrame({
            "feature": churn_features[:len(churn_model.feature_importances_)],
            "importance": churn_model.feature_importances_
        }).sort_values("importance", ascending=False)
        feature_importance["churn_feature_importance"] = churn_importance
    else:
        feature_importance["churn_feature_importance"] = pd.DataFrame(
            columns=["feature", "importance"]
        )

    # -------------------------------
    # LTV feature importance
    # -------------------------------
    if ltv_model is not None and hasattr(ltv_model, "feature_importances_"):
        ltv_importance = pd.DataFrame({
            "feature": ltv_features[:len(ltv_model.feature_importances_)],
            "importance": ltv_model.feature_importances_
        }).sort_values("importance", ascending=False)
        feature_importance["ltv_feature_importance"] = ltv_importance
    else:
        feature_importance["ltv_feature_importance"] = pd.DataFrame(
            columns=["feature", "importance"]
        )

    return feature_importance

# =========================================================================
# PDF REPORT GENERATION
# =========================================================================

def fig_to_image(fig, width=450):
    """Convert Plotly figure to ReportLab Image"""
    if fig is None:
        return None
        
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format="png", scale=2)
    img_buffer.seek(0)
    return Image(img_buffer, width=width, height=width * 0.6)

@st.cache_data
def generate_ecommerce_pdf(kbi, yearly_orders_profit, figures, top_10_volume_drivers, 
                          payment_failure_analysis, churn_report, ltv_metrics):
    """Generate comprehensive PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>E-Commerce Analytics Report</b>", styles["Title"]))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    # KBI
    elements.append(Paragraph("<b>Key Business Indicators (KBI)</b>", styles["Heading2"]))
    kbi_table = [["Metric", "Value"]] + kbi.astype(str).values.tolist()
    table = Table(kbi_table, colWidths=[3.5 * inch, 2.2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E4057")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 12),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Yearly Performance
    elements.append(Paragraph("<b>Yearly Revenue & Profit</b>", styles["Heading2"]))
    yearly_table = [["Year", "Total Orders", "Profit (USD)"]] + yearly_orders_profit.astype(str).values.tolist()
    table = Table(yearly_table, colWidths=[1.5 * inch, 1.8 * inch, 2.2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F618D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # Charts
    elements.append(Paragraph("<b>Key Analytics Visuals</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    for title, fig in figures.items():
        if fig is not None:
            elements.append(Paragraph(f"<b>{title}</b>", styles["Normal"]))
            elements.append(Spacer(1, 6))
            img = fig_to_image(fig)
            if img:
                elements.append(img)
            elements.append(Spacer(1, 20))

    # Top Products
    elements.append(Paragraph("<b>Top 10 Revenue Drivers</b>", styles["Heading2"]))
    if not top_10_volume_drivers.empty:
        product_table = [["Product", "Revenue", "Cost", "Profit", "Margin"]] + \
                       top_10_volume_drivers.head(10).astype(str).values.tolist()
        table = Table(product_table, repeatRows=1, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#145A32")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No product data available.", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Payment Risk
    elements.append(Paragraph("<b>Payment Failure Risk</b>", styles["Heading2"]))
    if not payment_failure_analysis.empty:
        payment_table = [["Method", "Failed", "Total", "Failure Rate (%)"]] + \
                       payment_failure_analysis.astype(str).values.tolist()
        table = Table(payment_table, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#922B21")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No payment failure data available.", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Model Performance
    elements.append(Paragraph("<b>Churn Model Performance</b>", styles["Heading2"]))
    
    if isinstance(churn_report, dict) and "1" in churn_report:
        elements.append(Paragraph(
            f"""
            <b>Precision:</b> {churn_report['1']['precision']:.2f}<br/>
            <b>Recall:</b> {churn_report['1']['recall']:.2f}<br/>
            <b>F1-Score:</b> {churn_report['1']['f1-score']:.2f}<br/><br/>
            <i>Explanation:</i> This churn model identifies customers likely to leave the platform.
            Higher recall ensures risky customers are not missed, while precision ensures marketing spend is not wasted.
            """,
            styles["Normal"]
        ))
    else:
        elements.append(Paragraph("Churn model metrics not available.", styles["Normal"]))
    
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Customer Lifetime Value (LTV) Model</b>", styles["Heading2"]))
    
    if isinstance(ltv_metrics, dict):
        elements.append(Paragraph(
            f"""
            <b>R² Score:</b> {ltv_metrics.get('r2', 'N/A')}<br/>
            <b>RMSE:</b> {ltv_metrics.get('rmse', 'N/A')}<br/><br/>
            <i>Explanation:</i> R² indicates how well the model explains customer spending behavior.
            RMSE reflects average prediction error in USD, guiding revenue forecasting decisions.
            """,
            styles["Normal"]
        ))
    else:
        elements.append(Paragraph("LTV model metrics not available.", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =========================================================================
# EMAIL FUNCTIONALITY
# =========================================================================

def send_ecommerce_email(pdf_buffer, recipient_email, kbi):
    """Send PDF report via email"""
    if not EMAIL_CONFIG["SENDER_EMAIL"] or not EMAIL_CONFIG["SENDER_PASSWORD"]:
        st.sidebar.error("Email credentials not configured. Set environment variables.")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_CONFIG["SENDER_EMAIL"]
        msg["To"] = recipient_email
        msg["Subject"] = f"E-Commerce Performance Report – {datetime.now().strftime('%d %B %Y')}"

        # Extract KPIs
        total_revenue = kbi.loc[
            kbi["Metric"] == "Total Revenue (USD)", "Value"
        ].values[0] if not kbi.empty else 0

        total_customers = kbi.loc[
            kbi["Metric"] == "Total Customers", "Value"
        ].values[0] if not kbi.empty else 0

        active_customer_pct = kbi.loc[
            kbi["Metric"] == "Active Customer Percentage (%)", "Value"
        ].values[0] if not kbi.empty else 0

        # Email body
        body = f"""
        <p>Hello,</p>
        <p>Please find attached the <strong>E-Commerce Performance Report (PDF)</strong>.</p>
        <ul>
            <li><strong>Total Revenue:</strong> ${float(total_revenue):,.2f}</li>
            <li><strong>Total Customers:</strong> {int(total_customers):,}</li>
            <li><strong>Active Customers:</strong> {float(active_customer_pct):.2f}%</li>
        </ul>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p>Regards,<br/><strong>Analytics Team</strong></p>
        """
        msg.attach(MIMEText(body, "html"))

        # Attach PDF
        pdf_buffer.seek(0)
        part = MIMEBase("application", "pdf")
        part.set_payload(pdf_buffer.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", 'attachment; filename="Ecommerce_Report.pdf"')
        msg.attach(part)

        # Send email
        server = smtplib.SMTP(EMAIL_CONFIG["SMTP_SERVER"], EMAIL_CONFIG["SMTP_PORT"])
        server.starttls()
        server.login(EMAIL_CONFIG["SENDER_EMAIL"], EMAIL_CONFIG["SENDER_PASSWORD"])
        server.send_message(msg)
        server.quit()
        return True
        
    except Exception as e:
        st.sidebar.error(f"Failed to send email: {str(e)}")
        return False

# =========================================================================
# STREAMLIT UI
# =========================================================================

def main():
    """Main Streamlit app"""
    st.title("🛒 E-Commerce Analytics, Churn & LTV Intelligence Platform")
    st.markdown("An end-to-end e-commerce data application combining ETL, analytics, machine learning, and explainability.")

    # Sidebar
    st.sidebar.header("🗄️ Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Use hosted database (auto-download)", "Upload database file"],
        index=0
    )

    # Get database path
    raw_db_path = None
    if data_source == "Use hosted database (auto-download)":
        raw_db_path = download_and_validate_database(GOOGLE_DRIVE_SHARE_URL, LOCAL_DB_PATH)
        st.sidebar.success("✅ Database ready")
    else:
        uploaded_db = st.sidebar.file_uploader("Upload database", type=["db", "sqlite"])
        if uploaded_db:
            with open("uploaded_temp.db", "wb") as f:
                f.write(uploaded_db.getbuffer())
            raw_db_path = "uploaded_temp.db"
            if not _is_valid_sqlite_db(raw_db_path):
                st.error("❌ Uploaded file is not a valid SQLite database")
                st.stop()
            st.sidebar.success("✅ Database uploaded")
        else:
            st.sidebar.warning("👆 Please upload a database file")
            st.stop()

    # Run ETL
    with st.spinner("🔄 Running ETL pipeline..."):
        cleaned_data = run_etl_pipeline(raw_db_path, save_to_sqlite=False)

    st.success("✅ ETL completed! Data is cached for this session.")

    # Data Preview
    st.subheader("🔍 Data Preview (Top 5 Rows)")
    tables = {
        "Orders": cleaned_data["orders"],
        "Customers": cleaned_data["customers"],
        "Products": cleaned_data["products"],
        "Reviews": cleaned_data["reviews"],
    }

    cols = st.columns(len(tables))
    for col, (table_name, df) in zip(cols, tables.items()):
        with col:
            st.markdown(f"**{table_name}**")
            st.dataframe(df.head(5), use_container_width=True)
            
            st.download_button(
                label=f"⬇️ Download {table_name} CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{table_name.lower()}_cleaned.csv",
                mime="text/csv",
                key=f"download_{table_name}"
            )

    # Analytics Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Customers & Products",
        "📦 Orders, Payments & Logistics",
        "🤖 Churn & LTV Modeling",
        "🧠 Feature Importance & Explainability"
    ])

    # Prepare common data
    kbi = compute_kbi(cleaned_data)
    yearly_orders_profit = compute_yearly_orders_profit(cleaned_data)
    
    # Prepare figures
    figures_for_pdf = {}
    
    # Tab 1: Customers & Products
    with tab1:
        st.subheader("👥 Customer Engagement & Churn Overview")

        # Monthly Active Customers
        st.markdown("### 📈 Monthly Active Customers")
        try:
            mac_df, mac_fig = monthly_active_customers_analytics(cleaned_data)
            st.plotly_chart(mac_fig, use_container_width=True)
            figures_for_pdf["Monthly Active Customers"] = mac_fig
        except Exception as e:
            st.error(f"Error in monthly active customers: {e}")
        st.divider()

        # Monthly Churn Rate
        st.markdown("### 🔄 Monthly Churn Overview")
        try:
            churn_df = monthly_churn_rate(cleaned_data["orders"])
            churn_bar_fig = plot_monthly_churn(churn_df)
            st.plotly_chart(churn_bar_fig, use_container_width=True)
            figures_for_pdf["Monthly Churn Rate"] = churn_bar_fig
        except Exception as e:
            st.error(f"Error in monthly churn: {e}")
        st.divider()

        # Customer Churn Timeline
        st.markdown("### 🕒 Customer Churn Timeline")
        try:
            churn_flag_df = customer_churn_flag(cleaned_data["orders"])
            if len(churn_flag_df) > 100:
                churn_flag_df = churn_flag_df.sample(n=100, random_state=42)
            churn_scatter_fig = plot_customer_churn_scatter(churn_flag_df)
            st.plotly_chart(churn_scatter_fig, use_container_width=True)
            figures_for_pdf["Customer Churn Timeline"] = churn_scatter_fig
        except Exception as e:
            st.error(f"Error in churn timeline: {e}")
        st.divider()

        # Churn Drivers
        st.markdown("### 📉 Churn Drivers — Loyalty Score vs Age")
        try:
            churn_attr_fig = churn_customer_churn_fig(cleaned_data, sample_size=200)
            st.plotly_chart(churn_attr_fig, use_container_width=True)
            figures_for_pdf["Churn Drivers"] = churn_attr_fig
        except Exception as e:
            st.error(f"Error in churn drivers: {e}")

        # Revenue by Country
        st.subheader("🌍 Revenue Distribution by Country")
        try:
            rev_country_df = revenue_by_country_analytics(cleaned_data, top_n=None)
            if not rev_country_df.empty:
                fig_country_map = revenue_by_country_map(cleaned_data)
                st.plotly_chart(fig_country_map, use_container_width=True)
                figures_for_pdf["Revenue by Country"] = fig_country_map
            else:
                st.warning("No revenue data available for country analysis.")
        except Exception as e:
            st.error(f"Error in revenue by country: {e}")

        # Loyalty Analysis
        st.subheader("💎 Loyalty Impact on Customer Value")
        try:
            loyalty_df = loyalty_analysis_analytics(cleaned_data, top_n=200)
            if not loyalty_df.empty:
                fig_loyalty = plot_loyalty_analysis(loyalty_df)
                st.plotly_chart(fig_loyalty, use_container_width=True)
                figures_for_pdf["Loyalty Analysis"] = fig_loyalty
            else:
                st.warning("Insufficient data for loyalty analysis.")
        except Exception as e:
            st.error(f"Error in loyalty analysis: {e}")

        # Volume Drivers
        st.subheader("📦 Top Revenue-Driving Products")
        try:
            volume_drivers_df = volume_driver_analysis(cleaned_data, top_n=50)
            if not volume_drivers_df.empty:
                fig_volume = plot_volume_drivers(volume_drivers_df, top_n=10)
                st.plotly_chart(fig_volume, use_container_width=True)
                figures_for_pdf["Top Products"] = fig_volume
            else:
                st.warning("Product volume data unavailable.")
        except Exception as e:
            st.error(f"Error in volume drivers: {e}")

    # Tab 2: Orders, Payments & Logistics
    with tab2:
        st.subheader("🚚 Delivery Performance & Customer Experience")

        # Delivery Time vs Rating
        try:
            delivery_df = delivery_time_rating_analysis(cleaned_data, limit=200)
            fig_delivery = plot_delivery_time_vs_rating(delivery_df)
            if fig_delivery:
                st.plotly_chart(fig_delivery, use_container_width=True)
                figures_for_pdf["Delivery Time vs Rating"] = fig_delivery
        except Exception as e:
            st.error(f"Error in delivery analysis: {e}")
        st.divider()

        # Delivery Rate by Warehouse
        try:
            warehouse_df = delivery_rate_by_warehouse(cleaned_data)
            fig_warehouse = plot_delivery_rate_by_warehouse(warehouse_df)
            if fig_warehouse:
                st.plotly_chart(fig_warehouse, use_container_width=True)
                figures_for_pdf["Delivery Rate by Warehouse"] = fig_warehouse
        except Exception as e:
            st.error(f"Error in warehouse analysis: {e}")
        st.divider()

        # Payment Failure Rate
        try:
            payment_df = payment_failure_rate_analysis(cleaned_data)
            fig_payment = plot_payment_failure_rate(payment_df)
            if fig_payment:
                st.plotly_chart(fig_payment, use_container_width=True)
                figures_for_pdf["Payment Failure Rate"] = fig_payment
        except Exception as e:
            st.error(f"Error in payment analysis: {e}")

        # Sales & Profit Trends
        st.subheader("💰 Sales, Profit & Order Trends")
        try:
            sales_profit_df = yearly_monthly_sales_profit_analysis(cleaned_data)

            # Monthly Sales & Profit Trend
            fig_sales_profit = plot_monthly_sales_profit_trend(sales_profit_df)
            st.plotly_chart(fig_sales_profit, use_container_width=True)
            figures_for_pdf["Sales & Profit Trend"] = fig_sales_profit

            st.divider()

            # Monthly Orders Trend
            fig_orders_trend = plot_monthly_orders_trend(sales_profit_df)
            st.plotly_chart(fig_orders_trend, use_container_width=True)
            figures_for_pdf["Orders Trend"] = fig_orders_trend
        except Exception as e:
            st.error(f"Error in sales/profit trends: {e}")

        # Currency & Spending
        st.subheader("🌍 Currency & Customer Spending Distribution")
        col1, col2 = st.columns(2)

        with col1:
            try:
                currency_df = currency_profit_sales_analysis(cleaned_data)
                fig_currency = plot_profit_by_currency(currency_df)
                if fig_currency:
                    st.plotly_chart(fig_currency, use_container_width=True)
                    figures_for_pdf["Profit by Currency"] = fig_currency
            except Exception as e:
                st.error(f"Error in currency analysis: {e}")

        with col2:
            try:
                spending_df = customer_spending_tier_analysis(cleaned_data)
                fig_spending = plot_customer_spending_tiers(spending_df)
                if fig_spending:
                    st.plotly_chart(fig_spending, use_container_width=True)
                    figures_for_pdf["Spending Tiers"] = fig_spending
            except Exception as e:
                st.error(f"Error in spending tiers: {e}")

    # Tab 3: ML Models
    with tab3:
        st.subheader("🤖 Customer Churn & Lifetime Value Predictions")
        st.markdown("""
        This section uses machine learning to:
        - Predict **customer churn risk**
        - Estimate **Customer Lifetime Value (LTV)**
        """)

        # Train models
        with st.spinner("Training churn & LTV models..."):
            model_results = train_churn_and_ltv_models(cleaned_data)
        
        churn_df = model_results["churn_df"]
        ltv_df = model_results["ltv_df"]
        churn_metrics = model_results["churn_metrics"]
        ltv_metrics = model_results["ltv_metrics"]
        
        st.success("✅ Models trained successfully")
        st.divider()

        # Model Performance
        st.subheader("📊 Model Performance Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            precision = churn_metrics.get("1", {}).get("precision", 0) if isinstance(churn_metrics, dict) else 0
            st.metric("Churn Model Precision (High Risk)", round(precision, 2))
        with col2:
            recall = churn_metrics.get("1", {}).get("recall", 0) if isinstance(churn_metrics, dict) else 0
            st.metric("Churn Model Recall (High Risk)", round(recall, 2))
        with col3:
            r2 = ltv_metrics.get("r2", 0) if isinstance(ltv_metrics, dict) else 0
            st.metric("LTV Model R² Score", round(r2, 3))

        st.divider()

        # Churn Predictions
        st.subheader("🚨 Churn Risk Predictions")
        try:
            churn_preview = (
                churn_df.sort_values("predicted_churn_prob", ascending=False)
                .loc[:, ["customer_id", "predicted_churn_prob", "churn_risk_category", 
                        "total_orders", "avg_order_value", "days_since_last_order", "loyalty_score"]]
                .head(100)
            )
            st.dataframe(churn_preview, use_container_width=True)
            st.caption("Top 100 customers ranked by predicted churn probability")
        except Exception as e:
            st.error(f"Error displaying churn predictions: {e}")
        st.divider()

        # LTV Predictions
        st.subheader("💰 Customer Lifetime Value (LTV) Predictions")
        try:
            ltv_preview = (
                ltv_df.sort_values("predicted_ltv_usd", ascending=False)
                .loc[:, ["customer_id", "predicted_ltv_usd", "total_orders", "avg_order_value", 
                        "customer_age_days", "loyalty_score"]]
                .head(100)
            )
            st.dataframe(ltv_preview, use_container_width=True)
            st.caption("Top 100 customers by predicted lifetime value")
        except Exception as e:
            st.error(f"Error displaying LTV predictions: {e}")

        st.divider()

        # Downloads
        st.subheader("⬇️ Download Prediction Outputs")
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download Churn Predictions (CSV)",
                churn_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                "Download LTV Predictions (CSV)",
                ltv_df.to_csv(index=False).encode("utf-8"),
                file_name="ltv_predictions.csv",
                mime="text/csv"
            )

    # Tab 4: Feature Importance
    with tab4:
        st.subheader("🧠 Feature Importance & Model Explainability")
        st.markdown("""
        This section explains **which features influence the models most**.
        Feature importance is extracted from the trained **Random Forest models**.
        """)

        # Get feature importance
        feature_importance = get_feature_importance(
            churn_model=model_results["churn_model"],
            ltv_model=model_results["ltv_model"],
            churn_features=["total_orders", "avg_order_value", "days_since_last_order", "loyalty_score", "account_age_days"],
            ltv_features=["total_orders", "avg_order_value", "days_since_last_order", "customer_age_days", 
                         "order_frequency_days", "loyalty_score"]
        )

        churn_fi = feature_importance["churn_feature_importance"]
        ltv_fi = feature_importance["ltv_feature_importance"]

        st.divider()

        # Churn Feature Importance
        st.subheader("🚨 Churn Model — Feature Importance")
        if not churn_fi.empty:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(churn_fi, use_container_width=True)

            with col2:
                fig_churn_fi = px.bar(
                    churn_fi,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Churn Feature Importance",
                    text="importance"
                )
                fig_churn_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_churn_fi.update_layout(yaxis=dict(categoryorder="total ascending"), height=450)
                st.plotly_chart(fig_churn_fi, use_container_width=True)
        else:
            st.warning("Churn feature importance is not available.")

        st.divider()

        # LTV Feature Importance
        st.subheader("💰 LTV Model — Feature Importance")
        if not ltv_fi.empty:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(ltv_fi, use_container_width=True)

            with col2:
                fig_ltv_fi = px.bar(
                    ltv_fi,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="LTV Feature Importance",
                    text="importance"
                )
                fig_ltv_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_ltv_fi.update_layout(yaxis=dict(categoryorder="total ascending"), height=450)
                st.plotly_chart(fig_ltv_fi, use_container_width=True)
        else:
            st.warning("LTV feature importance is not available.")

        st.divider()

        # Download buttons
        st.subheader("⬇️ Download Feature Importance")
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download Churn Feature Importance (CSV)",
                churn_fi.to_csv(index=False).encode("utf-8"),
                file_name="churn_feature_importance.csv",
                mime="text/csv"
            )

        with col2:
            st.download_button(
                "Download LTV Feature Importance (CSV)",
                ltv_fi.to_csv(index=False).encode("utf-8"),
                file_name="ltv_feature_importance.csv",
                mime="text/csv"
            )

    # =========================================================================
    # PDF REPORT & EMAIL
    # =========================================================================
    
    st.sidebar.header("📄 Report Export")
    
    # Prepare data for PDF
    volume_drivers_df = volume_driver_analysis(cleaned_data, top_n=50)
    top_10_volume_drivers = volume_drivers_df.head(10) if not volume_drivers_df.empty else pd.DataFrame()
    payment_failure_df = payment_failure_rate_analysis(cleaned_data)
    
    if st.sidebar.button("Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_buffer = generate_ecommerce_pdf(
                    kbi=kbi,
                    yearly_orders_profit=yearly_orders_profit,
                    figures=figures_for_pdf,
                    top_10_volume_drivers=top_10_volume_drivers,
                    payment_failure_analysis=payment_failure_df,
                    churn_report=churn_metrics,
                    ltv_metrics=ltv_metrics
                )
                st.session_state["pdf_buffer"] = pdf_buffer
                st.sidebar.success("✅ PDF generated successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to generate PDF: {str(e)}")

    if "pdf_buffer" in st.session_state:
        st.sidebar.download_button(
            label="⬇ Download PDF",
            data=st.session_state["pdf_buffer"],
            file_name="Ecommerce_Analytics_Report.pdf",
            mime="application/pdf"
        )

    # Email functionality
    st.sidebar.header("📧 Email Report")
    recipient_email = st.sidebar.text_input("Recipient Email", placeholder="example@email.com")

    if st.sidebar.button("Send PDF Report via Email", key="send_pdf_email"):
        if "pdf_buffer" not in st.session_state:
            st.sidebar.warning("⚠️ Please generate the PDF report first.")
        elif not recipient_email:
            st.sidebar.warning("⚠️ Please enter a recipient email.")
        else:
            with st.spinner("Sending email..."):
                success = send_ecommerce_email(
                    pdf_buffer=st.session_state["pdf_buffer"],
                    recipient_email=recipient_email,
                    kbi=kbi
                )
                if success:
                    st.sidebar.success("Email sent successfully ✅")
                else:
                    st.sidebar.error("Failed to send email. Check credentials.")

if __name__ == "__main__":
    main()
