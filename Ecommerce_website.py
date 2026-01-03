import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import ssl
from io import BytesIO
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate,Table,TableStyle,Paragraph,Spacer)
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score,mean_squared_error,mean_absolute_error)
from openpyxl import Workbook
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from openpyxl.styles import Font
import plotly.graph_objects as go
from reportlab.platypus import Image



import re
import unicodedata
import plotly.express as px
import os
from tqdm import tqdm
import pycountry
import random
import string
from datetime import datetime, timedelta
import requests
import warnings

# ---------------------- Utilities ----------------------
if "db_conn" not in st.session_state:
    st.session_state.db_conn = None

if "tables_loaded" not in st.session_state:
    st.session_state.tables_loaded = False

if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = {} 

# ---------------------- App Metadata ----------------------
APP_TITLE = "🛒 E-Commerce Analytics & ML Dashboard"
APP_LAYOUT = "wide"

SQLITE_DB_PATH = "Ecommerce.db"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = st.secrets.get("SENDER_EMAIL", "jimmyukaba1234@gmail.com")
SENDER_PASSWORD = st.secrets.get("SENDER_PW", "ldww szmx lmnh eoks")
DEFAULT_RECIPIENTS = (
    st.secrets.get("RECIPIENTS", "").split(",")
    if st.secrets.get("RECIPIENTS")
    else []
)
# 1. ETL PIPELINE WITH ML PREDICTION CAPABILITIES
def clean_ecommerce_dataset(
    sqlite_db_path: str ="Ecommerce.db",
    run_tables=("customers", "orders", "products", "reviews"),
    save_to_sqlite: bool = False
    ):
    """
    ETL pipeline for E-commerce Dataset.
    """
    print("=" * 80)
    print("ECOMMERCE DATASET CLEANING PIPELINE")
    print("=" * 80)

    # ------------------- CONFIG ------------------------------------
    warnings.filterwarnings("ignore")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    # ------------------- CHECK DATABASE EXISTS ----------------------
    if not Path(sqlite_db_path).exists():
        raise FileNotFoundError(f"Database file not found: {sqlite_db_path}")

    # ------------------- CONNECTION SETUP --------------------------
    conn = sqlite3.connect(sqlite_db_path)

    try:
        # ------------------- EXTRACTION --------------------------------
        print("\n1. EXTRACTING DATA...")
        print("=" * 80)

        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        available_tables = [row[0] for row in cursor.fetchall()]

        tables_to_load = [t for t in run_tables if t in available_tables]

        if not tables_to_load:
            raise RuntimeError(f"No requested tables found. Available: {available_tables}")

        dataframes = {}
        for table in tables_to_load:
            df = pd.read_sql_query(f"SELECT * FROM [{table}]", conn)  # [] to handle reserved names
            dataframes[table] = df
            print(f"Loaded {table}: {df.shape}")

        # --------------------------------- 2. TRANSFORMATION PHASE --------------------------
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
                first = row["first_name"]
                last = row["last_name"]
                if first != "Unknown" and last != "Unknown":
                    return f"{first} {last}"
                elif first != "Unknown":
                    return first
                elif last != "Unknown":
                    return last
                else:
                    return "Unknown"

            df["full_name"] = df.apply(build_full_name, axis=1)

            #-------------COUNTRY---------------------------------
            if "country" in df.columns:
                df["country"] = df["country"].str.title().fillna("Unknown")

                # COUNTRY STANDARDIZATION
                def standardize_country(country):
                    if country == "Unknown":
                        return "Unknown"
                    try:
                        return pycountry.countries.lookup(country).name
                    except LookupError:
                        # Try partial match or common variants
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
                    {
                        "Unknown": np.nan,
                        "unknown": np.nan,
                        "": np.nan,
                        pd.NA: np.nan
                        }
                    )

                df["account_age_days"] = pd.to_numeric(df["account_age_days"], errors="coerce")

                    # 2. Convert negative values to positive
                df["account_age_days"] = df["account_age_days"].abs()

                valid_vals = df["account_age_days"].dropna()

                if not valid_vals.empty:
                    rng = np.random.default_rng(seed=42)
                    min_val = valid_vals.min()
                    max_val = valid_vals.max()

                    nan_mask = df["account_age_days"].isna()
                    df.loc[nan_mask, "account_age_days"] = rng.integers(
                        low=int(min_val),
                        high=int(max_val) + 1,
                        size=nan_mask.sum()
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
                        df[col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .replace(["", "nan", "none", "null"], np.nan)
                        .map(boolean_map)
                    )
            # Credit tier--
            if "credit_tier" in df.columns:
                df["credit_tier"] = (df["credit_tier"].astype(str).str.strip().str.upper())

            # Preferred language
            if "preferred_language" in df.columns:
                language_map = {"ESPAÑOL": "ES", "SPANISH": "ES", "ENGLISH": "EN", "NONE": "EN"}
                df["preferred_language"] = (
                    df["preferred_language"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace(language_map)
                    .replace(["", "NAN", "NONE", "NULL",], np.nan)
                    .fillna("EN")
                )

            # Currency preference
            if "currency_preference" in df.columns:
                currency_map = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY"}
                df["currency_preference"] = (
                    df["currency_preference"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace(currency_map)
                    .replace(['  ',"", "NAN", "NONE", "NULL"], np.nan)
                )

            # Customer status
            if "customer_status" in df.columns:
                df["customer_status"] = (
                    df["customer_status"]
                    .astype(str)
                    .str.strip()
                    .str.title()
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
                    df["gender"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace(["", "nan", "none", "null"], np.nan)
                    .map(gender_map)
                    .fillna("Unknown")
                )

            # Numeric financial columns
            numeric_columns = ["total_spent", "avg_order_value", "loyalty_score"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            #drop tables
            drop_columns = ["email","phone","address","city","state","last_login","first_name","last_name","address_line1","last_login_time","last_login_date","zip_code","last_login_date"]
            df = df.drop(columns=[col for col in drop_columns if col in df.columns]) 

            #Droping duplicates and removing nan and unknown
            df = df.drop_duplicates()
            df = df[
                df["customer_id"].notna()
                & (df["customer_id"] != "Unknown")
                & df["country"].notna()
                & (df["country"] != "Unknown")
                & df["date_of_birth"].notna()
                & df["registration_date"].notna()
            ]

            # Numeric columns outliers, missing values and duplicate fix
            df["avg_order_value"]= df["avg_order_value"].abs()
            df["total_spent"]= df["total_spent"].abs()
            df["loyalty_score"]= df["loyalty_score"].abs()
            df["account_age_days"]= df["account_age_days"].abs()
            numeric_columns = ["account_age_days","loyalty_score","total_spent","avg_order_value"]
            for col in numeric_columns:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1

                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound),col] = np.nan
            rng = np.random.default_rng(seed=42)  # reproducible

            for col in numeric_columns:
                if col in df.columns:
                    valid_values = df[col].dropna()
                    # Safety check
                    if not valid_values.empty:
                        min_val = valid_values.min()
                        max_val = valid_values.max()
                        nan_mask = df[col].isna()
                        n_missing = nan_mask.sum()

                        df.loc[nan_mask, col] = rng.uniform(low=min_val,high=max_val,size=n_missing)
            #df["account_age_days"] = df["account_age_days"].round(0).astype("Int64")

            # =========================================================
            # CATEGORICAL COLUMNS — MODE IMPUTATION
            # =========================================================

            categorical_cols = ["customer_status","newsletter_subscribed","preferred_language","currency_preference","marketing_consent","gender","credit_tier"]
            for col in categorical_cols:
                if col in df.columns:

                    # Normalize placeholders
                    df[col] = df[col].replace({"Unknown": np.nan,"unknown": np.nan,"": np.nan,pd.NA: np.nan})
                    # Compute mode safely
                    if not df[col].dropna().empty:
                        mode_value = df[col].mode(dropna=True)[0]

                        # Fill missing with mode
                        df[col] = df[col].fillna(mode_value)

            print("Customers cleaning completed")
            cleaned_tables["customers"] = df

        # ----------------------------- ORDERS TABLE (example stub) -----------------------------
        if "orders" in dataframes:
            print("Cleaning orders table (basic)")
            df1 = dataframes["orders"].copy()

            id_columns = ["order_id", "customer_id", "product_id"]
            for col in id_columns:
                if col in df1.columns:
                    df1[col] = (df1[col].astype(str).str.strip()
                        .replace(["", "nan", "None", "NULL"], np.nan))
            #------Date columns------------------------
            date_columns = ["order_date","estimated_delivery","actual_delivery"]
            for col in date_columns:
                if col in df1.columns:
                    df1[col] = pd.to_datetime(df1[col],errors="coerce",infer_datetime_format=True)
            # 2. Split into separate columns
            df1["order_date_date"] = df1["order_date"].dt.date
            df1["order_date_time"] = df1["order_date"].dt.time

            # 3. Drop original column
            df1.drop(columns=["order_date"], inplace=True)

            #--------------FIXING NUMERIC COLUMNS-----------------
            numeric_columns = ["order_amount","quantity","shipping_cost","tax_amount","total_amount","discount_amount"]
            for col in numeric_columns:
                if col in df1.columns:
                    # Remove non-numeric characters (keep minus & decimal)
                    df1[col] = (df1[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True))
                    df1[col] = pd.to_numeric(df1[col], errors="coerce")
                    # Convert negatives to positives
                    df1[col] = df1[col].abs()
                    # Rounding rules
                    if col == "quantity":
                        df1[col] = df1[col].round(0)
                    else:
                        df1[col] = df1[col].round(3)

            # ORDER CATEGORICAL COLUMNS — NORMALIZATION + MODE IMPUTATION
            
            # Helper to normalize missing placeholders
            def _normalize_missing(series):
                return series.replace(
                    {
                        "": np.nan,
                        "nan": np.nan,
                        "NaN": np.nan,
                        "None": np.nan,
                        "NONE": np.nan,
                        "null": np.nan,
                        "NULL": np.nan,
                        "Unknown": np.nan,
                        "UNKNOWN": np.nan,
                        pd.NA: np.nan
                    }
                )

            # -------------------------
            # -------------------------
            # PAYMENT METHOD (TITLE CASE ONLY)
            # -------------------------
            if "payment_method" in df1.columns:
                df1["payment_method"] = (
                    df1["payment_method"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace({
                        "CREDIT CARD": "CREDIT CARD",
                        "CREDIT_CARD": "CREDIT CARD",
                        "CARD": "CREDIT CARD",
                        "PAYPAL": "PAYPAL",
                        "APPLE PAY": "APPLE PAY",
                        "APPLE_PAY": "APPLE PAY",
                        "GOOGLE PAY": "GOOGLE PAY",
                        "GOOGLE_PAY": "GOOGLE PAY",
                        "BANK TRANSFER": "BANK TRANSFER",
                        "BANK_TRANSFER": "BANK TRANSFER",
                        "CASH": "CASH"
                    })
                    .pipe(_normalize_missing)
                )

                # Final format → Title Case ONLY
                df1["payment_method"] = df1["payment_method"].str.title()

                # Mode imputation
                if not df1["payment_method"].dropna().empty:
                    mode_payment_method = df1["payment_method"].mode(dropna=True)[0]
                    df1["payment_method"] = df1["payment_method"].fillna(mode_payment_method)

            # -------------------------
            # SHIPPING METHOD
            # -------------------------
            if "shipping_method" in df1.columns:df1["shipping_method"] = (df1["shipping_method"].astype(str).str.strip().str.title().pipe(_normalize_missing))
            # Mode imputation
            if not df1["shipping_method"].dropna().empty:
                mode_shipping = df1["shipping_method"].mode(dropna=True)[0]
                df1["shipping_method"] = df1["shipping_method"].fillna(mode_shipping)

            # -------------------------
            # ORDER STATUS
            # -------------------------
            if "order_status" in df1.columns:df1["order_status"] = (df1["order_status"].astype(str).str.strip().str.title().pipe(_normalize_missing))
            # Mode imputation
            if not df1["order_status"].dropna().empty:
                mode_order_status = df1["order_status"].mode(dropna=True)[0]
                df1["order_status"] = df1["order_status"].fillna(mode_order_status)

            # -------------------------
            # -------------------------
            # CURRENCY CLEANING
            # -------------------------
            if "currency" in df1.columns:
                currency_map = {
                    "USD": "USD",
                    "EUR": "EUR",
                    "GBP": "GBP"
                }

                df1["currency"] = (
                    df1["currency"]
                    .astype(str)
                    .str.strip()                      # remove whitespace
                    .replace({"": np.nan, "None": np.nan})  # empty → NaN
                    .str.upper()
                    .map(currency_map)
                )

                # Fill missing currencies with mode
                mode_currency = df1["currency"].mode(dropna=True)[0]
                df1["currency"] = df1["currency"].fillna(mode_currency)

            # =========================================================
            # WAREHOUSE — MODE IMPUTATION (NO CASE CHANGE)
            # =========================================================

            if "warehouse_id" in df1.columns:
    
                # 1. Normalize missing placeholders
                df1["warehouse_id"] = (
                    df1["warehouse_id"].astype(str).str.strip().replace(
                        {
                            "": np.nan,
                            "nan": np.nan,
                            "NaN": np.nan,
                            "None": np.nan,
                            "null": np.nan,
                            "NULL": np.nan,
                            "Unknown": np.nan,
                            "UNKNOWN": np.nan,
                            pd.NA: np.nan
                        }
                    )
                )

                # 2. Fill missing with mode
                if not df1["warehouse_id"].dropna().empty:
                    mode_warehouse = df1["warehouse_id"].mode(dropna=True)[0]
                    df1["warehouse_id"] = df1["warehouse_id"].fillna(mode_warehouse)
                
            # =========================================================
            #        CHANNEL — TITLE CASE + MODE IMPUTATION
            # =========================================================

            if "channel" in df1.columns:

                # 1. Normalize placeholders
                df1["channel"] = (df1["channel"].astype(str).str.strip()
                    .replace(
                        {
                            "": np.nan,
                            "nan": np.nan,
                            "NaN": np.nan,
                            "None": np.nan,
                            "NONE": np.nan,
                            pd.NA: np.nan
                        }
                    )
                )

                # 2. Convert to Title Case
                df1["channel"] = df1["channel"].str.title()

                # 3. Fill missing with mode
                if not df1["channel"].dropna().empty:
                    mode_channel = df1["channel"].mode(dropna=True)[0]
                    df1["channel"] = df1["channel"].fillna(mode_channel)
            
            def get_exchange_rates(base="USD"):
                url = f"https://open.er-api.com/v6/latest/{base}"
                response = requests.get(url)
                response.raise_for_status()
                return response.json()["rates"]

            rates = get_exchange_rates("USD")
            order_money_columns = ["order_amount","shipping_cost","tax_amount","total_amount","discount_amount"]

            for col in order_money_columns:
                if col in df1.columns:
                    df1[f"{col}_usd"] = df1.apply(
                        lambda row: (
                            row[col] / rates[row["currency"]]
                            if pd.notna(row[col]) and row["currency"] in rates
                            else np.nan
                        ),
                        axis=1
                    )

            # -----------------------------------------------------------------------
            #                DROP UNUSED / UNWANTED COLUMNS (UPDATED)
            # -----------------------------------------------------------------------
            drop_columns = ["shipping_address","sales_rep_id","return_reason","discount_code","notes","shipping_address_same"]
            df1 = df1.drop(columns=[col for col in drop_columns if col in df1.columns])

            cleaned_tables["orders"] = df1
            print("Orders cleaning completed")

        # -----------------------------------------------------------------------------------------------
        # ========================================REVIEWS TABLE==========================================
        # -----------------------------------------------------------------------------------------------
        if "reviews" in dataframes:
            print("Cleaning reviews table (basic)")
            df2 = dataframes["reviews"].copy()

            # ID COLUMNS — NORMALIZE MISSING VALUES
            
            id_columns = ["review_id","customer_id","product_id","order_id"]
            for col in id_columns:
                if col in df2.columns:
                    df2[col] = ( df2[col].astype(str).str.strip()
                        .replace(
                            {
                                "": np.nan,
                                pd.NA: np.nan
                            }
                        )
                    ) 
            #-------DATES COLUMN-----------------
            date_columns = ["review_date"]
            for col in date_columns:
                if col in df2.columns: 
                    df2[col] = pd.to_datetime(df2[col],errors="coerce")

            # NUMERIC COLUMNS — ABSOLUTE VALUES + OUTLIER REMOVAL

            numeric_cols = ["rating","value_for_money","helpful","unhelpful"]
            for col in numeric_cols:
                if col in df2.columns:
                    df2[col] = pd.to_numeric(df2[col], errors="coerce")
                    df2[col] = df2[col].abs()

                    # 3. Remove outliers using IQR (set to NaN, do not drop rows)
                    q1 = df2[col].quantile(0.25)
                    q3 = df2[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr

                    df2.loc[(df2[col] < lower) | (df2[col] > upper),col] = np.nan
            
            # REVIEWS — CATEGORICAL & BOOLEAN NORMALIZATION
            def _normalize_missing(s):
                return s.replace(
                to_replace=["", "none", "nan", "null", "n/a", "na"],
                value=np.nan,
                regex=False
            )

            # LANGUAGE (UPPER CASE + STANDARDIZATION)
            if "language" in df2.columns:
                df2["language"] = (df2["language"].astype(str).str.strip().pipe(_normalize_missing).str.upper()
                    .replace({"ENGLISH": "EN",  "NONE": np.nan,"NAN": np.nan}
                    )
                )

                if not df2["language"].dropna().empty:
                    mode_language = df2["language"].mode(dropna=True)[0]
                    df2["language"] = df2["language"].fillna(mode_language)
            
            # BOOLEAN COLUMNS — NORMALIZATION + MODE IMPUTATION

            boolean_columns = ["verified_purchase","would_recommend"]
            boolean_map = {
                "1": True,"Y": True,
                "YES": True,"TRUE": True,
                "0": False,"N": False,
                "NO": False,"FALSE": False
            }

            for col in boolean_columns:
                if col in df2.columns:

                    df2[col] = (df2[col].astype(str).str.strip().str.upper().replace(boolean_map)
                        .replace(
                            {
                                "": np.nan,
                                "NaN": np.nan,
                                "None": np.nan,
                                "NONE": np.nan
                            }
                        )
                    )

                    # Mode imputation
                    if not df2[col].dropna().empty:
                        mode_val = df2[col].mode(dropna=True)[0]
                        df2[col] = df2[col].fillna(mode_val)

            # REVIEW STATUS (TITLE CASE + MODE)
            if "review_status" in df2.columns:
                df2["review_status"] = (df2["review_status"].astype(str).str.strip().pipe(_normalize_missing).str.title())
                if not df2["review_status"].dropna().empty:
                    mode_review_status = df2["review_status"].mode(dropna=True)[0]
                    df2["review_status"] = df2["review_status"].fillna(mode_review_status)

            if "product_condition" in df2.columns:
                df2["product_condition"] = (df2["product_condition"]
                    .str.strip()
                    .pipe(_normalize_missing)   # convert None, "", "nan" → NaN
            )

            if not df2["product_condition"].dropna().empty:
                mode_condition = df2["product_condition"].mode(dropna=True)[0]
                df2["product_condition"] = df2["product_condition"].fillna(mode_condition)

            # Format LAST
            df2["product_condition"] = df2["product_condition"].str.title()

            columns_to_drop = ["response_from_seller","response_date","delivery_rating","reviewer_expertise","review_text","review_title"]
            df2 = df2.drop(columns=[col for col in columns_to_drop if col in df2.columns])

            print("reviews cleaning completed")
            cleaned_tables["reviews"] = df2

        # ----------------------------- PRODUCT TABLE -----------------------------
        if "products" in dataframes:
            print("Cleaning products table")

            df3 = dataframes["products"].copy()
            df3["product_id"] = (
                df3["product_id"].astype(str).str.strip()
                .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                .fillna("Unknown")
            )

            # SKU
            df3["sku"] = df3["sku"].astype(str).str.strip()
            df3["sku"] = df3["sku"].replace(["", "nan", "none", "None", "NONE", "null"], np.nan)

            # Function to generate one SKU in format: SKU-12345-ABC
            def generate_sku():
                # Random 5-digit number (reproducible)
                num_part = random.randint(10000, 99999)
                
                # Random 3-letter uppercase code (reproducible)
                letter_part = ''.join(random.choices(string.ascii_letters, k=3))
                
                return f"SKU-{num_part}-{letter_part}"

            # Apply only to missing SKUs
            df3["sku"] = df3["sku"].apply(lambda x: generate_sku() if pd.isna(x) else x)

            # Ensure final SKU column is string type
            df3["sku"] = df3["sku"].astype(str)

            # CATEGORIES
            for col in ["main_category", "sub_category"]:
                df3[col] = (
                    df3[col].astype(str).str.strip().str.title()
                    .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                )

            # BRAND
            df3["brand"] = (
                df3["brand"].astype(str).str.strip().str.title()
                .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                .fillna("Unknown")
            )

            # TAX CATEGORY
            df3["tax_category"] = (
                df3["tax_category"].astype(str).str.strip().str.title()
                .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                #.fillna("Unknown")
            )

            # PRODUCT STATUS
            df3["product_status"] = (
                df3["product_status"].astype(str).str.strip().str.title()
                .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
                #.fillna("Unknown")
            )

            # THE BOOLS
            bool_map = {
                "true": True,
                "false": False,
                "yes": True,
                "no": False,
                "Y": True,
                "N": False,
                "y": True,
                "n": False,
                "1": True,
                "0": False,
                True: True,
                False: False
            }

            df3["is_digital"] = (
                df3["is_digital"].astype(str).str.strip().str.title()
                .replace(bool_map)
                .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
            )

            df3["requires_shipping"] = (
                df3["requires_shipping"].astype(str).str.strip().str.title()
                .replace(bool_map)
                .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
            )

            # CURRENCY STANDARDIZATION
            currency_map = {"$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR",
                            "gbp": "GBP", "£": "GBP", "¥": "JPY", "jpy": "JPY"}

            df3["currency"] = (
                df3["currency"].astype(str).str.strip().str.upper()
                .replace(currency_map)
                .replace(["", "nan", "none", "None", "NONE", "null"], np.nan)
            )
            # STOCK QUANTITY, WEIGHT, REVIEWS, WARRANTY MONTHS
            df3["stock_quantity"] = pd.to_numeric(df3["stock_quantity"], errors="coerce")
            df3["weight_kg"] = pd.to_numeric(df3["weight_kg"], errors="coerce")
            df3["review_count"] = pd.to_numeric(df3["review_count"], errors="coerce")
            df3["warranty_months"] = pd.to_numeric(df3["warranty_months"], errors="coerce")

            # CONVERT NEGATIVES → ABSOLUTE VALUES
            for col in ["price", "cost", "stock_quantity", "warranty_months"]:
                if col in df3.columns:
                    df3[col] = df3[col].abs()

            # DROP UNWANTED COLUMNS
            drop_cols = [
                "product_name", "sub_category", "dimensions", "color",
                "size", "material", "supplier_id", "description", "tags", "manufacturer"
            ]
            df3 = df3.drop(columns=[c for c in drop_cols if c in df3.columns])

            # OUTLIER CLIPPING (Upper IQR)
            num_cols = ["price", "cost", "stock_quantity"]
            def clip_upper_iqr(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                upper = Q3 + 1.5 * IQR
                return series.apply(lambda x: upper if (pd.notna(x) and x > upper) else x)

            for col in num_cols:
                df3[col] = clip_upper_iqr(df3[col])

            df3["stock_quantity"] = df3["stock_quantity"].round(0).astype("Int64")

            # RANDOM VALUES WITHIN COLUMN RANGE (for missing)
            def fill_random_within_range(series):
                non_null = series.dropna()
                if non_null.empty:
                    return series.fillna(0)
                low, high = non_null.min(), non_null.max()
                return series.apply(
                    lambda x: np.random.uniform(low, high) if pd.isna(x) else x
                )

            for col in ["stock_quantity", "weight_kg", "review_count"]:
                if col in df3.columns:
                    df3[col] = fill_random_within_range(df3[col])

            # COST & PRICE GENERATION (Numeric-Safe Version)
            df3["cost"] = pd.to_numeric(df3["cost"], errors="coerce")
            df3["price"] = pd.to_numeric(df3["price"], errors="coerce")

            cost_min, cost_max = df3["cost"].min(), df3["cost"].max()

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

            # WARRANTY → multiples of 12, max 60
            df3["warranty_months"] = df3["warranty_months"].apply(
                lambda x: np.random.choice([12, 24, 36, 48, 60]) if pd.isna(x) else x
            )

            # COUNTRY OF ORIGIN STANDARDIZATION
            def standardize_country(val):
                if pd.isna(val):
                    return np.nan
                try:
                    return pycountry.countries.lookup(val).name
                except:
                    return val

            if "country_of_origin" in df3.columns:
                df3["country_of_origin"] = df3["country_of_origin"].astype(str).str.strip()
                df3["country_of_origin"] = df3["country_of_origin"].replace(
                    ["", "nan", "none", "None", "NONE", "null"], np.nan
                )
                df3["country_of_origin"] = df3["country_of_origin"].apply(standardize_country)

            # RANDOM DATE FILLING
            for col in ["created_date", "last_updated"]:
                if col not in df3.columns:
                    continue
            df3[col] = pd.to_datetime(df3[col], errors="coerce")
            # MODE IMPUTATION
            mode_cols = ["main_category", "currency", "product_status", "country_of_origin", "is_digital",
                "requires_shipping", "tax_category"]

            for col in mode_cols:
                if col in df3.columns:
                    mode_val = df3[col].mode(dropna=True)[0] if df3[col].notna().any() else "Unknown"
                    df3[col] = df3[col].fillna(mode_val)

            # RATING → int + random 1–5
            df3["rating"] = pd.to_numeric(df3["rating"], errors="coerce")
            df3["rating"] = df3["rating"].apply(lambda x: int(x) if pd.notna(x) else np.nan)
            df3["rating"] = df3["rating"].apply(
                lambda x: np.random.randint(1, 6) if pd.isna(x) else x
            )

            print("products cleaning completed")
            cleaned_tables["products"] = df3
        

        # ------------------- SAVE CLEANED TABLES BACK TO SQLITE -------------
        if save_to_sqlite:
            print("\nSaving cleaned tables back to database...")
            for table_name, df_clean in cleaned_tables.items():
                df_clean.to_sql(table_name + "_clean", conn, if_exists="replace", index=False)
                print(f"Saved {table_name}_clean with shape {df_clean.shape}")

        print("\nPipeline completed successfully!")

        return cleaned_tables

    finally:
        conn.close()


# ------------------- RUN THE PIPELINE -------------------
if __name__ == "__main__":
    # Adjust path if your DB is not in the current directory
    cleaned_data = clean_ecommerce_dataset(
        sqlite_db_path=r"C:\Users\DELL\E Commerce2\Ecommerce.db",           # Change if needed
        run_tables=("customers", "orders", "products", "reviews"),      # Add more tables as needed
        save_to_sqlite=True                      # Set to True to save cleaned versions
    )


#=============================ANALYTICS AND VISUALIZATION========================================
def monthly_active_customers_analytics(cleaned_data):
    df2 = cleaned_data["orders"]

    monthly_active = (df2
        .dropna(subset=["order_date_date"])
        .assign(
            year_month=lambda x: pd.to_datetime(x["order_date_date"]).dt.to_period("M").astype(str)
        )
        .groupby("year_month", as_index=False)
        .agg(active_customers=("customer_id", "nunique"))
        .sort_values("year_month")
    )

    # ------------- Visualization ------------
    fig = px.line(
        monthly_active,
        x="year_month",
        y="active_customers",
        title="Monthly Active Customers Over Time",
        labels={
            "year_month": "Year-Month",
            "active_customers": "Active Customers"
        },
        markers=True
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Active Customers",
        template="plotly_white"
    )
    return monthly_active, fig

#------------------Customer churn by loyalty score and age------------------------
def churn_customer_attributes(cleaned_data, sample_size=100):
    df_customers = cleaned_data["customers"]
    df_orders = cleaned_data["orders"]

    monthly_customers = (
        df_orders[["customer_id", "order_date_date"]]
        .dropna()
        .assign(
            year_month=lambda x: pd.to_datetime(x["order_date_date"])
            .dt.to_period("M")
            .astype(str)
        )
        .drop_duplicates()
        )

    monthly_customers["next_month"] = (
        pd.to_datetime(monthly_customers["year_month"] + "-01")
        + pd.offsets.MonthBegin(1)
    ).dt.to_period("M").astype(str)

    churn_flag = monthly_customers.merge(
        monthly_customers[["customer_id", "year_month"]],
        left_on=["customer_id", "next_month"],
        right_on=["customer_id", "year_month"],
        how="left",
        indicator=True
        )

    churn_flag["churned"] = (churn_flag["_merge"] == "left_only").astype(int)

    churn_flag = churn_flag.rename(
        columns={"year_month_x": "year_month"}
    )[["customer_id", "year_month", "churned"]]

    customers_attr = df_customers.copy()
    customers_attr["date_of_birth"] = pd.to_datetime(
        customers_attr["date_of_birth"], errors="coerce"
    )

    customers_attr["age"] = (
        (pd.Timestamp("now") - customers_attr["date_of_birth"]).dt.days / 365.25
    )
    customers_attr["age"] = pd.to_numeric(
        customers_attr["age"], errors="coerce"
    ).round(0)

    churn_attributes = churn_flag.merge(
        customers_attr[
            ["customer_id", "country", "credit_tier", "loyalty_score",
             "age", "newsletter_subscribed", "marketing_consent"]
        ],
        on="customer_id",
        how="left"
    ).dropna(subset=["age"])

    return churn_attributes.head(sample_size)


def churn_customer_churn_fig(cleaned_data, sample_size=100):
    
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
#-----------Monthly churn------------------
def monthly_churn_rate(df2):
    orders = (df2
        .dropna(subset=["order_date_date", "customer_id"]).assign(
            year_month=lambda x: pd.to_datetime(x["order_date_date"])
            .dt.to_period("M")
            .astype(str))
        [["customer_id", "year_month"]]
        .drop_duplicates())
    # Shift to get next month
    next_month = orders.copy()
    next_month["year_month"] = (
        pd.to_datetime(next_month["year_month"] + "-01")
        + pd.DateOffset(months=1)
    ).dt.to_period("M").astype(str)

    merged = orders.merge(next_month,on=["customer_id", "year_month"],how="left",indicator=True)
    churn_df = (merged
        .groupby("year_month", as_index=False).agg(
            active_customers=("customer_id", "nunique"),
            churned_customers=("_merge", lambda x: (x == "left_only").sum())
        )
    )

    churn_df["churn_rate"] = (churn_df["churned_customers"] / churn_df["active_customers"]).round(4)
    return churn_df.sort_values("year_month")

def plot_monthly_churn(churn_df):
    fig = px.bar(churn_df,
        x="year_month",
        y=["active_customers", "churned_customers"],
        barmode="group",
        title="Active vs Churned Customers per Month",
        labels={
            "value": "Number of Customers",
            "year_month": "Month",
            "variable": "Customer Type"
        },
        color_discrete_map={
            "active_customers": "green",
            "churned_customers": "red"
        },
        height=500
    )
    fig.update_yaxes(range=[0, churn_df["active_customers"].max() * 1.1])
    return fig
# -----------------------CHURN FLAG----------------------------
def customer_churn_flag(df_orders, churn_threshold_months=3):
    
    df = df_orders.copy()

    # Ensure datetime
    df["order_date_date"] = pd.to_datetime(
        df["order_date_date"],
        errors="coerce"
    )

    df = df.dropna(subset=["order_date_date"])

    # Create year-month
    df["year_month"] = df["order_date_date"].dt.to_period("M").astype(str)

    # Last purchase per customer
    last_purchase = (
        df.groupby("customer_id")["order_date_date"]
        .max()
        .reset_index(name="last_order_date")
    )

    # Reference date (latest order in dataset)
    reference_date = df["order_date_date"].max()

    # Months inactive
    last_purchase["months_inactive"] = (
        (reference_date - last_purchase["last_order_date"])
        / pd.Timedelta(days=30)
    )

    # Churn flag
    last_purchase["churned"] = (
        last_purchase["months_inactive"] >= churn_threshold_months
    ).astype(int)

    # Merge churn flag back to monthly view
    churn_df = (
        df[["customer_id", "year_month"]]
        .drop_duplicates()
        .merge(
            last_purchase[["customer_id", "churned"]],
            on="customer_id",
            how="left"
        )
    )
    return churn_df

def plot_customer_churn_scatter(churn_flag):
    fig_scatter = px.scatter(
        churn_flag.assign(churned_str=churn_flag["churned"].astype(str)),
        x="customer_id",
        y="year_month",
        color="churned_str",
        title="Customer Churn Over Time",
        labels={
            "customer_id": "Customer ID",
            "year_month": "Year-Month",
            "churned_str": "Churned"
        },
        color_discrete_map={"0": "green", "1": "red"}
    )

    fig_scatter.update_yaxes(categoryorder="category ascending")
    return fig_scatter

# -------------Revenue by country-----------------------------
def revenue_by_country_analytics(cleaned_data, top_n=50):
    """
    Revenue and customer count by country.
    """
    df_customers = cleaned_data["customers"]  # customers_clean

    revenue_by_country = (
        df_customers
        .groupby("country", as_index=False)
        .agg(
            customer_count=("customer_id", "nunique"),
            revenue=("total_spent", "sum")
        )
        .sort_values("revenue", ascending=False)
        .head(top_n)
    )

    return revenue_by_country
def revenue_by_country_map(cleaned_data, top_n=None):
    revenue_by_country = revenue_by_country_analytics(
        cleaned_data=cleaned_data,
        top_n=top_n
    )

    fig_map = px.choropleth(
        revenue_by_country,
        locations="country",              # ✅ REQUIRED
        locationmode="country names",     # ✅ REQUIRED
        color="revenue",
        hover_name="country",
        color_continuous_scale="Viridis",
        title="Revenue by Country"
    )

    fig_map.update_layout(  
        margin=dict(l=20, r=20, t=60, b=20),
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="natural earth"
        )
    )

    return fig_map

#------------------------------Loyalty score--------------------------
def loyalty_analysis_analytics(cleaned_data, top_n=100):
    """
    Analyze top 10% customers by total_spent and compute
    purchase frequency and lifetime value.
    """

    df_customers = cleaned_data["customers"]   # customers_clean
    df_orders = cleaned_data["orders"]          # orders_clean

    # -------------------------------------------------
    # 1. Rank customers by total_spent (PERCENT_RANK equivalent)
    # -------------------------------------------------
    customers_ranked = df_customers.copy()

    customers_ranked["rank"] = (
        customers_ranked["total_spent"]
        .rank(method="average", pct=True)
    )

    # Top 10% customers
    top_customers = customers_ranked[
        customers_ranked["rank"] >= 0.9
    ]

    # -------------------------------------------------
    # 2. Join orders with top customers
    # -------------------------------------------------
    merged = df_orders.merge(
        top_customers,
        on="customer_id",
        how="inner"
    )

    # -------------------------------------------------
    # 3. Aggregate metrics
    # -------------------------------------------------
    loyalty_analysis = (
        merged
        .groupby(
            ["customer_id", "country", "credit_tier", "loyalty_score", "newsletter_subscribed"],
            as_index=False
        )
        .agg(
            purchase_frequency=("order_id", "count"),
            lifetime_value=("total_amount", "sum")
        )
        .sort_values("lifetime_value", ascending=False)
        .head(top_n)
    )

    return loyalty_analysis

def plot_loyalty_analysis(loyalty_analysis):
    fig = px.scatter(
        loyalty_analysis,
        x="purchase_frequency",
        y="lifetime_value",
        color="credit_tier",             # premium vs regular
        size="loyalty_score",            # bigger = more loyal
        symbol="newsletter_subscribed",  # newsletter behavior
        hover_data=["customer_id", "country"],
        title="Customer Purchase Frequency vs Lifetime Value"
    )

    fig.update_layout(
        xaxis_title="Purchase Frequency",
        yaxis_title="Lifetime Value (LTV)",
        height=450
    )
    return fig

#------------Volume drivers----------------------------------
def volume_driver_analysis(cleaned_data, top_n=100):
    df_products = cleaned_data["products"]
    df_orders = cleaned_data["orders"]

    # Merge orders with products
    df = df_orders.merge(
        df_products,
        on="product_id",
        how="left"
    )

    # Aggregate metrics per product
    volume_drivers = (
        df.groupby(["product_id"], as_index=False)
        .agg(
            revenue=("total_amount", "sum"),
            cost=("quantity", lambda x: (x * df.loc[x.index, "cost"]).sum())
        )
    )

    # Profit & margin
    volume_drivers["profit"] = volume_drivers["revenue"] - volume_drivers["cost"]
    volume_drivers["margin"] = volume_drivers["profit"] / volume_drivers["revenue"]

    # Sort by revenue (like SQL)
    volume_drivers = volume_drivers.sort_values(
        "revenue", ascending=False
    ).head(top_n)

    return volume_drivers

def plot_volume_drivers(volume_drivers_df, top_n=10):
    """
    Bar chart highlighting Volume Drivers vs Profit Engines.
    """
    top_revenue = volume_drivers_df.head(top_n)

    fig = px.bar(
        top_revenue,
        x="revenue",
        y="product_id",
        orientation="h",
        color="margin",
        text="profit",
        title="Top 10 Products by Revenue (Volume Drivers vs Profit Engines)",
        labels={
            "revenue": "Revenue (USD)",
            "product_id": "Product",
            "margin": "Profit Margin"
        },
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=450
    )
    return fig

#---------------------------Buttom products ------------------------
def bottom_products_analysis(cleaned_data, bottom_pct=0.05, max_rows=100):
    """
    Identify bottom-performing products based on low revenue and low ratings.
    Equivalent to the SQL CTE + ranking logic.
    """

    df_products = cleaned_data["products"]   # products_clean
    df_orders = cleaned_data["orders"]       # orders_clean
    df_reviews = cleaned_data["reviews"]     # reviews_clean

    # -------------------------------------------------
    # 1. Revenue per product
    # -------------------------------------------------
    product_sales = (
        df_orders
        .groupby("product_id", as_index=False)
        .agg(revenue=("total_amount", "sum"))
    )

    # -------------------------------------------------
    # 2. Average rating per product
    # -------------------------------------------------
    product_ratings = (
        df_reviews
        .groupby("product_id", as_index=False)
        .agg(avg_rating=("rating", "mean"))
    )

    # -------------------------------------------------
    # 3. Merge revenue + ratings
    # -------------------------------------------------
    product_perf = (
        product_sales
        .merge(product_ratings, on="product_id", how="inner")
        .dropna()
    )

    # -------------------------------------------------
    # 4. Rank products (low revenue + low rating)
    # -------------------------------------------------
    product_perf = product_perf.sort_values(
        ["revenue", "avg_rating"],
        ascending=[True, True]
    ).reset_index(drop=True)

    total_products = len(product_perf)
    cutoff = max(1, int(bottom_pct * total_products))

    bottom_products = product_perf.head(min(cutoff, max_rows))

    # -------------------------------------------------
    # 5. Visualization
    # -------------------------------------------------
    fig = px.scatter(
        bottom_products,
        x="avg_rating",
        y="revenue",
        size="revenue",
        color="avg_rating",
        hover_name="product_id",
        title="Bottom Performing Products: Revenue vs Average Rating",
        labels={
            "avg_rating": "Average Rating",
            "revenue": "Revenue (USD)"
        },
        color_continuous_scale="RdYlGn_r"
    )

    fig.update_layout(
        height=450
    )

    return bottom_products, fig

def compute_kbi(cleaned_data):
    customers = cleaned_data["customers"]
    orders = cleaned_data["orders"]

    active_statuses = ["active", "gold", "silver", "premium"]
    inactive_statuses = ["inactive", "suspended"]

    total_customers = customers["customer_id"].nunique()

    active_customers = customers.loc[
        customers["customer_status"].str.lower().isin(active_statuses),
        "customer_id"
    ].nunique()

    inactive_customers = customers.loc[
        customers["customer_status"].str.lower().isin(inactive_statuses),
        "customer_id"
    ].nunique()

    unknown_status_customers = total_customers - (active_customers + inactive_customers)

    total_orders = orders["order_id"].nunique()
    total_revenue = round(orders["total_amount_usd"].sum(), 2)
    avg_order_value = round(orders["total_amount_usd"].mean(), 2)
    avg_orders_per_customer = round(total_orders / total_customers, 2)
    active_customer_pct = round((active_customers / total_customers) * 100, 2)

    kbi = pd.DataFrame({
        "Metric": [
            "Total Customers",
            "Active Customers",
            "Inactive Customers",
            "Customers with Unknown Status",
            "Active Customer Percentage (%)",
            "Total Orders",
            "Total Revenue (USD)",
            "Average Order Value (USD)",
            "Average Orders per Customer"
        ],
        "Value": [
            total_customers,
            active_customers,
            inactive_customers,
            unknown_status_customers,
            active_customer_pct,
            total_orders,
            total_revenue,
            avg_order_value,
            avg_orders_per_customer
        ]
    })

    return kbi

#===================================================================================================================================
# Delivery time rating
def delivery_time_rating_analysis(cleaned_data, limit=200):
    """
    Analyze relationship between delivery time and product rating.
    """
    df_products = cleaned_data["products"]   # products_clean
    df_orders = cleaned_data["orders"]       # orders_clean

    delivery_rating_df = (
        df_orders.merge(
            df_products[["product_id", "rating"]],
            on="product_id",
            how="inner"
        )
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

def plot_delivery_time_vs_rating(df):
    """
    Scatter plot: Delivery Time vs Product Rating
    """
    fig = px.scatter(
        df,
        x="delivery_time_days",
        y="rating",
        trendline="ols",
        title="Relationship Between Delivery Time and Product Rating",
        labels={
            "delivery_time_days": "Delivery Time (Days)",
            "rating": "Product Rating"
        }
    )

    return fig
#-------------------Delivery by warehouse-------------------------
def delivery_rate_by_warehouse(cleaned_data):
    """
    Calculate delivery rate per warehouse.
    Delivery rate = shipped_orders / total_orders * 100
    """
    df_orders = cleaned_data["orders"].copy()

    analytics_df = (
        df_orders
        .groupby("warehouse_id", as_index=False)
        .agg(
            shipped_orders=("actual_delivery", lambda x: x.notna().sum()),
            total_orders=("order_id", "nunique")
        )
    )

    analytics_df["delivery_rate"] = (
        analytics_df["shipped_orders"] / analytics_df["total_orders"] * 100
    ).round(2)

    return analytics_df

def plot_delivery_rate_by_warehouse(analytics_df):
    fig = px.bar(
        analytics_df,
        x="warehouse_id",
        y="delivery_rate",
        title="Delivery Rate by Warehouse",
        labels={
            "warehouse_id": "Warehouse ID",
            "delivery_rate": "Delivery Rate (%)"
        }
    )

    fig.update_traces(
        texttemplate="%{y:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_ticksuffix="%",
        xaxis_tickangle=-30,
        height=450
    )

    return fig
#-----------------Payment failure-----------------------
def payment_failure_rate_analysis(cleaned_data):
    """
    Calculate failed payment rate by payment method.
    """
    orders_df = cleaned_data["orders"].copy()

    orders_df["payment_status"] = (
        orders_df["payment_status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    payment_failure_df = (
        orders_df
        .groupby("payment_method", as_index=False)
        .agg(
            all_payments=("order_id", "count"),
            failed_payments=("payment_status", lambda x: (x == "failed").sum())
        )
    )

    payment_failure_df["failed_payment_rate"] = (
        100
        * payment_failure_df["failed_payments"]
        / payment_failure_df["all_payments"].replace(0, np.nan)
    ).round(2)
    payment_failure_df = payment_failure_df.dropna(
        subset=["failed_payment_rate"]
    ).sort_values("failed_payment_rate", ascending=False)

    return payment_failure_df

def plot_payment_failure_rate(payment_df):
    """
    Bar chart for failed payment rate by payment method.
    """
    fig = px.bar(
        payment_df,
        x="payment_method",
        y="failed_payment_rate",
        title="Failed Payment Rate by Payment Method",
        labels={
            "payment_method": "Payment Method",
            "failed_payment_rate": "Failed Payment Rate (%)"
        },
        text="failed_payment_rate")
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside")
    fig.update_layout(
        yaxis_ticksuffix="%",
        xaxis_tickangle=-30,
        height=450,
        yaxis_range=[0, payment_df["failed_payment_rate"].max() * 1.2])

    return fig

@st.cache_data
def compute_yearly_orders_profit(cleaned_data):
    df2 = cleaned_data["orders"].copy()
    df1 = cleaned_data["products"].copy()

    # Ensure datetime
    df2["order_date_date"] = pd.to_datetime(
        df2["order_date_date"],
        errors="coerce"
    )

    # Merge orders + products ONCE
    orders_products = df2.merge(
        df1,
        on="product_id",
        how="inner"
    )

    # Row-level profit
    orders_products["profit_usd"] = (
        orders_products["total_amount_usd"]
        - (orders_products["cost"] * orders_products["quantity"])
    )

    # Yearly aggregation
    yearly_orders_profit = (
        orders_products
        .assign(order_year=orders_products["order_date_date"].dt.year)
        .groupby("order_year", as_index=False)
        .agg(
            total_orders=("order_id", "count"),
            total_profit_usd=("profit_usd", "sum")
        )
        .sort_values("order_year")
    )

    return yearly_orders_profit
yearly_orders_profit = compute_yearly_orders_profit(cleaned_data)
def yearly_monthly_sales_profit_analysis(cleaned_data):
    """
    Compute yearly & monthly sales, total orders, and profit (USD-adjusted).
    """
    df_orders = cleaned_data["orders"]
    df_products = cleaned_data["products"]

    df = (
        df_orders
        .merge(
            df_products[["product_id", "price", "cost"]],
            on="product_id",
            how="inner"
        )
        .dropna(subset=["order_date_date"])
    )

    df["order_date_date"] = pd.to_datetime(df["order_date_date"])
    df["order_year"] = df["order_date_date"].dt.year.astype(str)
    df["order_month"] = df["order_date_date"].dt.month.astype(str).str.zfill(2)

    # Profit with currency normalization
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

def plot_monthly_sales_profit_trend(df_year_month):
    """
    Interactive monthly sales & profit trend with year dropdown (fixed visibility logic).
    """
    fig = go.Figure()

    years = sorted(df_year_month["order_year"].unique())
    n_years = len(years)

    # Add traces (2 per year: Sales + Profit)
    for idx, year in enumerate(years):
        df_year = df_year_month[df_year_month["order_year"] == year]

        # Sales trace
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

        # Profit trace
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

    # Dropdown buttons (visibility control)
    buttons = []

    for idx, year in enumerate(years):
        visible = [False] * (n_years * 2)

        visible[idx * 2] = True       # Sales
        visible[idx * 2 + 1] = True   # Profit

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

    # Layout + dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.02,
                y=1.15,
            )
        ],
        title=f"Monthly Sales & Profit Trend – {years[0]}",
        xaxis=dict(title="Month"),
        yaxis=dict(
            title="Sales (USD)",
            tickprefix="$"
        ),
        yaxis2=dict(
            title="Profit (USD)",
            overlaying="y",
            side="right",
            tickprefix="$"
        ),
        height=450,
        legend_title="Metric",
        template="plotly_white"
    )

    return fig


    return fig

def plot_monthly_orders_trend(df_year_month):
    """
    Interactive monthly orders trend with year dropdown.
    Expects columns:
    - order_year
    - order_month
    - total_orders
    """
    fig = go.Figure()

    years = sorted(df_year_month["order_year"].unique())

    # One trace per year
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


#--------------------Currency profit sales-----------------------------
def currency_profit_sales_analysis(cleaned_data):
    """
    Aggregate total sales and profit by currency.
    """
    df_orders = cleaned_data["orders"]     # orders_clean
    df_products = cleaned_data["products"] # products_clean

    df = (
        df_orders.merge(
            df_products[["product_id", "price", "cost"]],
            on="product_id",
            how="inner"
        )
        .assign(
            profit_usd=lambda x: np.where(
                x["currency"] == "USD", (x["price"] - x["cost"]),
                np.where(
                    x["currency"] == "EUR", (x["price"] - x["cost"]) * 1.09,
                    np.where(
                        x["currency"] == "GBP", (x["price"] - x["cost"]) * 1.27,
                        0
                    )
                )
            )
        )
        .groupby("currency", as_index=False)
        .agg(
            total_sales_usd=("total_amount_usd", "sum"),
            total_profit_usd=("profit_usd", "sum")
        )
        .round(2)
        .sort_values(
            ["total_profit_usd", "total_sales_usd"],
            ascending=False
        )
    )

    return df

def plot_profit_by_currency(currency_df):
    """
    Donut chart showing profit contribution by currency.
    """
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

    fig.update_layout(
        height=450
    )
    return fig

#----------------------------Customer spending tier----------------------------
def customer_spending_tier_analysis(cleaned_data):
    """
    Analyze customer distribution by spending tier.
    """
    df_customers = cleaned_data["customers"].copy()

    # Ensure numeric
    df_customers = df_customers.dropna(subset=["total_spent"])

    # Create spending tiers (same logic as SQL CASE)
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
        df_customers
        .groupby("spending_tier", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            avg_spent=("total_spent", "mean"),
            avg_account_age=("account_age_days", "mean")
        )
        .round({
            "avg_spent": 2,
            "avg_account_age": 1
        })
        .sort_values("avg_spent", ascending=False)
    )

    return spending_summary

def plot_customer_spending_tiers(spending_df):
    """
    Pie chart showing customer distribution by spending tier.
    """
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


#================================================================================================================================
#------------------Churn and Customer Lifetime Value (LTV) Model Prediction--------------------------
def train_churn_and_ltv_models(cleaned_data, reference_date="2025-12-30"):
    np.random.seed(42)
    # Currency rates
    currency_rates = {"USD": 1.0,"EUR": 1.1,"GBP": 1.25}

    # Prepare orders
    orders = cleaned_data["orders"].copy()
    orders = orders[~orders["order_status"].isin(["cancelled", "returned"])]

    orders["order_date_date"] = pd.to_datetime(orders["order_date_date"], errors="coerce")
    orders["total_amount"] = pd.to_numeric(
        orders["total_amount"], errors="coerce"
    ).fillna(0)

    orders["total_amount_usd"] = orders.apply(
        lambda x: x["total_amount"] * currency_rates.get(x["currency"], 1.0),
        axis=1
    )

    orders["customer_id"] = orders["customer_id"].astype(str)

    customers = cleaned_data["customers"].copy()
    customers["customer_id"] = customers["customer_id"].astype(str)

    today = pd.Timestamp(reference_date)

    # =====================================================
    # ======================= CHURN =======================
    # =====================================================

    customer_orders = (
        orders
        .groupby("customer_id")
        .agg(
            last_order_date=("order_date_date", "max"),
            total_orders=("order_id", "count"),
            avg_order_value=("total_amount_usd", "mean")
        )
        .reset_index()
    )

    df_churn = customers.merge(
        customer_orders, on="customer_id", how="left"
    )

    # --------- SAFE COLUMN RESOLUTION ----------
    if "avg_order_value" not in df_churn.columns:
        if "avg_order_value_y" in df_churn.columns:
            df_churn["avg_order_value"] = df_churn["avg_order_value_y"]
        elif "avg_order_value_x" in df_churn.columns:
            df_churn["avg_order_value"] = df_churn["avg_order_value_x"]
        else:
            df_churn["avg_order_value"] = 0

    if "total_orders" not in df_churn.columns:
        df_churn["total_orders"] = 0

    # --------- SAFE FILLS ----------
    df_churn["total_orders"] = df_churn["total_orders"].fillna(0)
    df_churn["avg_order_value"] = df_churn["avg_order_value"].fillna(0)

    df_churn["last_order_date"] = pd.to_datetime(
        df_churn["last_order_date"], errors="coerce"
    ).fillna(today)

    df_churn["days_since_last_order"] = (
        today - df_churn["last_order_date"]
    ).dt.days

    # --------- RFM SCORING ----------
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

    df_churn["target"] = (
        df_churn["churn_risk_category"] == "High Risk"
    ).astype(int)

    churn_features = [
        "total_orders",
        "avg_order_value",
        "days_since_last_order",
        "loyalty_score",
        "account_age_days"
    ]

    churn_features = [c for c in churn_features if c in df_churn.columns]

    X = df_churn[churn_features].fillna(0)
    y = df_churn["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    churn_model = RandomForestClassifier(
        n_estimators=200, random_state=42
    )
    churn_model.fit(X_train, y_train)

    churn_preds = churn_model.predict(X_test)

    df_churn["predicted_churn_prob"] = churn_model.predict_proba(X)[:, 1]

    churn_metrics = classification_report(
        y_test, churn_preds, output_dict=True
    )

    # =====================================================
    # ======================== LTV ========================
    # =====================================================

    customer_orders_ltv = (
        orders
        .groupby("customer_id")
        .agg(
            total_spent_usd=("total_amount_usd", "sum"),
            avg_order_value=("total_amount_usd", "mean"),
            total_orders=("order_id", "count"),
            last_order_date=("order_date_date", "max"),
            first_order_date=("order_date_date", "min")
        )
        .reset_index()
    )

    df_ltv = customers.merge(
        customer_orders_ltv, on="customer_id", how="left"
    )

    # --------- SAFE COLUMN HANDLING ----------
    for col in [
        "total_spent_usd",
        "avg_order_value",
        "total_orders"
    ]:
        if col not in df_ltv.columns:
            if f"{col}_y" in df_ltv.columns:
                df_ltv[col] = df_ltv[f"{col}_y"]
            elif f"{col}_x" in df_ltv.columns:
                df_ltv[col] = df_ltv[f"{col}_x"]
            else:
                df_ltv[col] = 0

    df_ltv["total_spent_usd"] = df_ltv["total_spent_usd"].fillna(0)
    df_ltv["avg_order_value"] = df_ltv["avg_order_value"].fillna(0)
    df_ltv["total_orders"] = df_ltv["total_orders"].fillna(0)

    df_ltv["last_order_date"] = pd.to_datetime(
        df_ltv["last_order_date"], errors="coerce"
    ).fillna(today)

    df_ltv["first_order_date"] = pd.to_datetime(
        df_ltv["first_order_date"], errors="coerce"
    ).fillna(today)

    df_ltv["days_since_last_order"] = (
        today - df_ltv["last_order_date"]
    ).dt.days

    df_ltv["customer_age_days"] = (
        today - df_ltv["first_order_date"]
    ).dt.days

    df_ltv["order_frequency_days"] = (
        df_ltv["customer_age_days"] /
        df_ltv["total_orders"].replace(0, np.nan)
    ).fillna(df_ltv["customer_age_days"])

    ltv_features = [
        "total_orders",
        "avg_order_value",
        "days_since_last_order",
        "customer_age_days",
        "order_frequency_days",
        "loyalty_score"
    ]

    ltv_features = [c for c in ltv_features if c in df_ltv.columns]

    X = df_ltv[ltv_features].fillna(0)
    y = df_ltv["total_spent_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ltv_model = RandomForestRegressor(
        n_estimators=200, random_state=42
    )
    ltv_model.fit(X_train, y_train)

    y_pred = ltv_model.predict(X_test)

    df_ltv["predicted_ltv_usd"] = ltv_model.predict(X)

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


# ---------------------------Feature importance-----------------------------------
def get_feature_importance(
    churn_model,
    ltv_model,
    churn_features,
    ltv_features
    ):
    """
    Extract feature importance for churn and LTV Random Forest models.
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


# PREPARE FIGURES FOR PDF
figures_for_pdf = {}

# --- Revenue Trend ---
sales_profit_df = yearly_monthly_sales_profit_analysis(cleaned_data)
fig_revenue_trend = plot_monthly_sales_profit_trend(sales_profit_df)
figures_for_pdf["Revenue Trend"] = fig_revenue_trend

# --- Customer Churn Timeline ---
churn_flag_df = customer_churn_flag(cleaned_data["orders"])
if len(churn_flag_df) > 100:
    churn_flag_df = churn_flag_df.sample(100, random_state=42)

fig_churn_timeline = plot_customer_churn_scatter(churn_flag_df)
figures_for_pdf["Customer Churn Timeline"] = fig_churn_timeline

# --- Payment Failure Rate ---
payment_df = payment_failure_rate_analysis(cleaned_data)
fig_payment_failure = plot_payment_failure_rate(payment_df)
figures_for_pdf["Payment Failure Rate"] = fig_payment_failure

# --- Revenue by Country ---
rev_country_df = revenue_by_country_analytics(cleaned_data)
if not rev_country_df.empty:
    fig_country_map = revenue_by_country_map(cleaned_data)
    figures_for_pdf["Top Countries by Revenue"] = fig_country_map

# Top 10 revenue-driving products (reuse the df you already created)
volume_drivers_df = volume_driver_analysis(cleaned_data, top_n=50)
if not volume_drivers_df.empty:
    top_10_volume_drivers = volume_drivers_df.head(10)
else:
    top_10_volume_drivers = pd.DataFrame(columns=["product_name", "revenue", "cost", "profit", "margin"])

churn_ltv_results = train_churn_and_ltv_models(cleaned_data)

churn_metrics = churn_ltv_results["churn_metrics"]
ltv_metrics = churn_ltv_results["ltv_metrics"]

#----------------------payment-----------------------------------------------------------------------------
payment_failure_df = payment_failure_rate_analysis(cleaned_data)
fig_payment_failure = plot_payment_failure_rate(payment_failure_df)
#==================================================================================================================================
#--------------------------------------------------------Streamlit UI phase--------------------------------------------------------
# APP CONFIG
st.set_page_config(
    page_title="🛒 E-Commerce Analytics, Churn & LTV Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 E-Commerce Analytics, Churn & LTV Intelligence Platform")
st.markdown(
    "An end-to-end e-commerce data application combining ETL, analytics, machine learning, and explainability."
)

# SIDEBAR: DATABASE SOURCE
st.sidebar.header("🗄️ Data Source")

data_source = st.sidebar.radio(
    "Choose data source",
    ["Use existing database", "Upload database (.db)"]
)

db_path = None

if data_source == "Use existing database":
    db_path = "Ecommerce.db"
    st.sidebar.success("✅ Using local Ecommerce.db")

else:
    uploaded_db = st.sidebar.file_uploader(
        "Upload Ecommerce Database",
        type=["db", "sqlite", "csv"]
    )

    if uploaded_db:
        db_path = "uploaded_ecommerce.db"
        with open(db_path, "wb") as f:
            f.write(uploaded_db.getbuffer())
        st.sidebar.success("✅ Database uploaded successfully")

if not db_path:
    st.info("👈 Select or upload a database to continue")
    st.stop()

@st.cache_data(show_spinner=False)
def load_cleaned_ecommerce_tables(db_path):
    conn = sqlite3.connect(db_path)

    tables = {
        "Orders": pd.read_sql("SELECT * FROM orders_clean", conn),
        "Customers": pd.read_sql("SELECT * FROM customers_clean", conn),
        "Products": pd.read_sql("SELECT * FROM products_clean", conn),
        "Reviews": pd.read_sql("SELECT * FROM reviews_clean", conn),
    }

    conn.close()
    return tables


with st.spinner("Loading data from database..."):
    tables = load_cleaned_ecommerce_tables(db_path)
st.success("✅ Data loaded successfully")
st.subheader("🔍 Data Preview (Top 5 Rows)")

cols = st.columns(len(tables))
for col, (table_name, df) in zip(cols, tables.items()):
    with col:
        st.markdown(f"### {table_name}")
        st.dataframe(df.head(5), use_container_width=True)

        # Download cleaned CSV
        st.download_button(
            label=f"⬇️ Download {table_name} CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{table_name.lower()}_cleaned.csv",
            mime="text/csv"
        )
st.session_state.cleaned_data = {
    "orders": tables["Orders"],
    "customers": tables["Customers"],
    "products": tables["Products"],
    "reviews": tables["Reviews"],
}
st.session_state.data_loaded = True

tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Customers & Products",
    "📦 Orders, Payments & Logistics",
    "🤖 Churn & LTV Modeling",
    "🧠 Feature Importance & Explainability"
    ])
with tab1:
    st.subheader("👥 Customer Engagement & Churn Overview")

    # 1️⃣ Monthly Active Customers

    st.markdown("### 📈 Monthly Active Customers")

    mac_df, mac_fig = monthly_active_customers_analytics(cleaned_data)
    st.plotly_chart(mac_fig, use_container_width=True)

    st.divider()

    # 2️⃣ Monthly Churn Rate (Active vs Churned)

    st.markdown("### 🔄 Monthly Churn Overview")

    churn_df = monthly_churn_rate(cleaned_data["orders"])
    churn_bar_fig = plot_monthly_churn(churn_df)
    st.plotly_chart(churn_bar_fig, use_container_width=True)

    st.divider()

    # 3️⃣ Customer Churn Timeline (Who churned & when)

    st.markdown("### 🕒 Customer Churn Timeline")

    churn_flag_df = customer_churn_flag(cleaned_data["orders"])

    # 🔹 Limit to 100 samples (stable random sample)
    if len(churn_flag_df) > 100:
        churn_flag_df = churn_flag_df.sample(
            n=100,
            random_state=42
        )

    churn_scatter_fig = plot_customer_churn_scatter(churn_flag_df)
    st.plotly_chart(churn_scatter_fig, use_container_width=True)

    st.divider()

    # Churn Drivers: Loyalty & Age
    st.markdown("### 📉 Churn Drivers — Loyalty Score vs Age")

    churn_attr_fig = churn_customer_churn_fig(
        cleaned_data,
        sample_size=200  # keeps UI fast
    )
    st.plotly_chart(churn_attr_fig, use_container_width=True)

    #----------------Revenue by Country--------------------
    st.set_page_config(layout="wide")
    st.subheader("🌍 Revenue Distribution by Country")
    rev_country_df = revenue_by_country_analytics(
        cleaned_data,
        top_n= None)
    if not rev_country_df.empty:
        st.plotly_chart(
            revenue_by_country_map(cleaned_data),
            use_container_width=True)
    else:
        st.warning("No revenue data available for country analysis.")

    # ----------------Loyalty and Lifetime value drivers--------------------
    st.subheader("💎 Loyalty Impact on Customer Value")
    loyalty_df = loyalty_analysis_analytics(
        cleaned_data,
        top_n=200)
    if not loyalty_df.empty:
        st.plotly_chart(
            plot_loyalty_analysis(loyalty_df),
            use_container_width=True)
    else:
        st.warning("Insufficient data for loyalty analysis.")

    # -----------------Top Revenue-Driving Products-------------------
    st.subheader("📦 Top Revenue-Driving Products")
    volume_drivers_df = volume_driver_analysis(
        cleaned_data,
        top_n=50)
    if not volume_drivers_df.empty:
        st.plotly_chart(
            plot_volume_drivers(volume_drivers_df, top_n=10),
            use_container_width=True)
    else:
        st.warning("Product volume data unavailable.")

with tab2:
    st.subheader("🚚 Delivery Performance & Customer Experience")

    # 1️⃣ Delivery Time vs Product Rating
    delivery_df = delivery_time_rating_analysis(cleaned_data, limit=200)
    st.plotly_chart(
        plot_delivery_time_vs_rating(delivery_df),
        use_container_width=True
    )

    st.divider()

    # 2️⃣ Delivery Rate by Warehouse
    warehouse_df = delivery_rate_by_warehouse(cleaned_data)
    st.plotly_chart(
        plot_delivery_rate_by_warehouse(warehouse_df),
        use_container_width=True
    )

    st.divider()

    # 3️⃣ Failed Payment Rate by Payment Method
    payment_df = payment_failure_rate_analysis(cleaned_data)
    st.plotly_chart(
        plot_payment_failure_rate(payment_df),
        use_container_width=True
    )

    st.subheader("💰 Sales, Profit & Order Trends")

    sales_profit_df = yearly_monthly_sales_profit_analysis(cleaned_data)

    # 4️⃣ Monthly Sales & Profit Trend
    st.plotly_chart(
        plot_monthly_sales_profit_trend(sales_profit_df),
        use_container_width=True
    )

    st.divider()

    # 5️⃣ Monthly Orders Trend
    st.plotly_chart(
        plot_monthly_orders_trend(sales_profit_df),
        use_container_width=True
    )

    st.subheader("🌍 Currency & Customer Spending Distribution")

    col1, col2 = st.columns(2)

    with col1:
        currency_df = currency_profit_sales_analysis(cleaned_data)
        st.plotly_chart(
            plot_profit_by_currency(currency_df),
            use_container_width=True
        )

    with col2:
        spending_df = customer_spending_tier_analysis(cleaned_data)
        st.plotly_chart(
            plot_customer_spending_tiers(spending_df),
            use_container_width=True
        )
#============================================================================================================================
with tab3:
    st.subheader("🤖 Customer Churn & Lifetime Value Predictions")

    st.markdown(
        """
        This section uses machine learning to:
        - Predict **customer churn risk**
        - Estimate **Customer Lifetime Value (LTV)**
        """
    )

    # Train models (cached)
    @st.cache_resource
    def run_models(cleaned_data):
        return train_churn_and_ltv_models(cleaned_data)

    with st.spinner("Training churn & LTV models..."):
        model_results = run_models(cleaned_data)

    churn_df = model_results["churn_df"]
    ltv_df = model_results["ltv_df"]
    churn_metrics = model_results["churn_metrics"]
    ltv_metrics = model_results["ltv_metrics"]
    st.success("✅ Models trained successfully")

    st.divider()

    # MODEL PERFORMANCE SUMMARY
    st.subheader("📊 Model Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Model Precision (High Risk)",
            round(churn_metrics["1"]["precision"], 2))
    with col2:
        st.metric(
            "Churn Model Recall (High Risk)",
            round(churn_metrics["1"]["recall"], 2))
    with col3:
        st.metric(
            "LTV Model R² Score",
            round(ltv_metrics["r2"], 3))
    st.divider()
    # CHURN PREDICTIONS
    st.subheader("🚨 Churn Risk Predictions")

    churn_preview = (
        churn_df
        .sort_values("predicted_churn_prob", ascending=False)
        .loc[:, [
            "customer_id",
            "predicted_churn_prob",
            "churn_risk_category",
            "total_orders",
            "avg_order_value",
            "days_since_last_order",
            "loyalty_score"
        ]]
        .head(100))
    st.dataframe(
        churn_preview,
        use_container_width=True)
    st.caption("Top 100 customers ranked by predicted churn probability")
    st.divider()

    # LTV PREDICTIONS
    st.subheader("💰 Customer Lifetime Value (LTV) Predictions")

    ltv_preview = (
        ltv_df
        .sort_values("predicted_ltv_usd", ascending=False)
        .loc[:, [
            "customer_id",
            "predicted_ltv_usd",
            "total_orders",
            "avg_order_value",
            "customer_age_days",
            "loyalty_score"
        ]]
        .head(100)
    )

    st.dataframe(
        ltv_preview,
        use_container_width=True
    )

    st.caption("Top 100 customers by predicted lifetime value")

    st.divider()

    # DOWNLOADS
    st.subheader("⬇️ Download Prediction Outputs")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download Churn Predictions (CSV)",
            churn_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv")
    with col2:
        st.download_button(
            "Download LTV Predictions (CSV)",
            ltv_df.to_csv(index=False).encode("utf-8"),
            file_name="ltv_predictions.csv",
            mime="text/csv")

with tab4:
    st.subheader("🧠 Feature Importance & Model Explainability")

    st.markdown(
        """
        This section explains **which features influence the models most**.
        Feature importance is extracted from the trained **Random Forest models**.
        """
    )
    # Get feature importance
    feature_importance = get_feature_importance(
        churn_model=model_results["churn_model"],
        ltv_model=model_results["ltv_model"],
        churn_features=[
            "total_orders",
            "avg_order_value",
            "days_since_last_order",
            "loyalty_score",
            "account_age_days"
        ],
        ltv_features=[
            "total_orders",
            "avg_order_value",
            "days_since_last_order",
            "customer_age_days",
            "order_frequency_days",
            "loyalty_score"
        ]
    )

    churn_fi = feature_importance["churn_feature_importance"]
    ltv_fi = feature_importance["ltv_feature_importance"]

    st.divider()
    # CHURN FEATURE IMPORTANCE
    st.subheader("🚨 Churn Model — Feature Importance")

    if not churn_fi.empty:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                churn_fi,
                use_container_width=True
            )

        with col2:
            fig_churn = px.bar(
                churn_fi,
                x="importance",
                y="feature",
                orientation="h",
                title="Churn Feature Importance",
                text="importance"
            )

            fig_churn.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_churn.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=450
            )

            st.plotly_chart(fig_churn, use_container_width=True)

    else:
        st.warning("Churn feature importance is not available.")

    st.divider()

    # LTV FEATURE IMPORTANCE
    st.subheader("💰 LTV Model — Feature Importance")

    if not ltv_fi.empty:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                ltv_fi,
                use_container_width=True
            )

        with col2:
            fig_ltv = px.bar(
                ltv_fi,
                x="importance",
                y="feature",
                orientation="h",
                title="LTV Feature Importance",
                text="importance"
            )

            fig_ltv.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            fig_ltv.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                height=450
            )

            st.plotly_chart(fig_ltv, use_container_width=True)

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
            mime="text/csv")

    with col2:
        st.download_button(
            "Download LTV Feature Importance (CSV)",
            ltv_fi.to_csv(index=False).encode("utf-8"),
            file_name="ltv_feature_importance.csv",
            mime="text/csv")
        
def fig_to_image(fig, width=450):
    """
    Convert a Plotly figure to a ReportLab Image.
    """
    img_buffer = BytesIO()

    # Convert Plotly fig → PNG
    fig.write_image(img_buffer, format="png", scale=2)

    img_buffer.seek(0)

    return Image(img_buffer, width=width, height=width * 0.6)
#-------------------------------PDF----------------------------------

kbi = compute_kbi(cleaned_data)
def generate_ecommerce_pdf(
    kbi,
    yearly_orders_profit,
    figures: dict,
    top_10_volume_drivers,
    payment_failure_analysis,
    churn_report,
    ltv_metrics
    ):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # ================= TITLE =================
    elements.append(Paragraph(
        "<b>E-Commerce Analytics Report</b>",
        styles["Title"]
    ))
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 14))

    # ================= KBI =================
    elements.append(Paragraph("<b>Key Business Indicators (KBI)</b>", styles["Heading2"]))

    kbi_table = [["Metric", "Value"]] + kbi.values.tolist()
    table = Table(kbi_table, colWidths=[3.5 * inch, 2.2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E4057")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ================= YEARLY PERFORMANCE =================
    elements.append(Paragraph("<b>Yearly Revenue & Profit</b>", styles["Heading2"]))

    yearly_table = [["Year", "Total Orders", "Profit (USD)"]] + yearly_orders_profit.values.tolist()
    table = Table(yearly_table, colWidths=[1.5 * inch, 1.8 * inch, 2.2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F618D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ================= CHARTS =================
    elements.append(Paragraph("<b>Key Analytics Visuals</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    for title, fig in figures.items():
        elements.append(Paragraph(f"<b>{title}</b>", styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(fig_to_image(fig))
        elements.append(Spacer(1, 20))

    # ================= TOP PRODUCTS =================
    elements.append(Paragraph("<b>Top 10 Revenue Drivers</b>", styles["Heading2"]))

    product_table = [["Product", "Revenue", "Cost", "Profit", "Margin"]] + top_10_volume_drivers.values.tolist()
    table = Table(product_table, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#145A32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ================= PAYMENT RISK =================
    elements.append(Paragraph("<b>Payment Failure Risk</b>", styles["Heading2"]))

    payment_table = [["Method", "Failed", "Total", "Failure Rate (%)"]] + payment_failure_analysis.values.tolist()
    table = Table(payment_table, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#922B21")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ================= MODEL PERFORMANCE =================
    elements.append(Paragraph("<b>Churn Model Performance</b>", styles["Heading2"]))

    elements.append(Paragraph(
        f"""
        <b>Precision:</b> {churn_report['1']['precision']:.2f}<br/>
        <b>Recall:</b> {churn_report['1']['recall']:.2f}<br/>
        <b>F1-Score:</b> {churn_report['1']['f1-score']:.2f}<br/><br/>
        <i>Explanation:</i>  
        This churn model identifies customers likely to leave the platform.  
        Higher recall ensures risky customers are not missed, while precision
        ensures marketing spend is not wasted on low-risk users.
        """,
        styles["Normal"]
    ))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Customer Lifetime Value (LTV) Model</b>", styles["Heading2"]))

    elements.append(Paragraph(
        f"""
        <b>R² Score:</b> {ltv_metrics['r2']}<br/>
        <b>RMSE:</b> {ltv_metrics['rmse']}<br/><br/>
        <i>Explanation:</i>  
        R² indicates how well the model explains customer spending behavior.  
        RMSE reflects average prediction error in USD, guiding revenue forecasting
        and customer segmentation decisions.
        """,
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
st.sidebar.header("📄 Report Export")
if st.sidebar.button("Generate PDF Report"):
    st.session_state["pdf_buffer"] = generate_ecommerce_pdf(
        kbi=kbi,
        yearly_orders_profit=yearly_orders_profit,
        figures={
            "Revenue Trend": fig_revenue_trend,
            "Customer Churn Timeline": fig_churn_timeline,
            "Payment Failure Rate": fig_payment_failure,
            "Top Countries by Revenue": fig_country_map,
        },
        top_10_volume_drivers=top_10_volume_drivers,
        payment_failure_analysis=payment_failure_df,
        churn_report=churn_metrics,
        ltv_metrics=ltv_metrics
    )
if "pdf_buffer" in st.session_state:
    st.sidebar.download_button(
        label="⬇ Download PDF",
        data=st.session_state["pdf_buffer"],
        file_name="Ecommerce_Analytics_Report.pdf",
        mime="application/pdf"
    )
EMAIL_SUBJECT = f"E-Commerce Performance Report – {datetime.now().strftime('%d %B %Y')}"
def send_ecommerce_email(pdf_buffer, recipient_email, kbi):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg["Subject"] = EMAIL_SUBJECT

    # -------- KPI extraction --------
    total_revenue = kbi.loc[
        kbi["Metric"] == "Total Revenue (USD)", "Value"
    ].values[0]

    total_customers = kbi.loc[
        kbi["Metric"] == "Total Customers", "Value"
    ].values[0]

    active_customer_pct = kbi.loc[
        kbi["Metric"] == "Active Customer Percentage (%)", "Value"
    ].values[0]

    # -------- Email body --------
    body = f"""
    <p>Hello,</p>

    <p>Please find attached the <strong>E-Commerce Performance Report (PDF)</strong>.</p>

    <ul>
        <li><strong>Total Revenue:</strong> ${total_revenue:,.2f}</li>
        <li><strong>Total Customers:</strong> {total_customers:,}</li>
        <li><strong>Active Customers:</strong> {active_customer_pct:.2f}%</li>
    </ul>

    <p>
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </p>

    <p>Regards,<br/>
    <strong>Analytics Team</strong></p>
    """

    msg.attach(MIMEText(body, "html"))

    # -------- Attach PDF from buffer --------
    pdf_buffer.seek(0)
    part = MIMEBase("application", "pdf")
    part.set_payload(pdf_buffer.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        'attachment; filename="Ecommerce_Report.pdf"'
    )
    msg.attach(part)

    # -------- Send email --------
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)
    server.quit()

st.sidebar.header("📧 Email Report")

recipient_email = st.sidebar.text_input(
    "Recipient Email",
    placeholder="example@email.com"
)

if st.sidebar.button("Send PDF Report via Email", key="send_pdf_email"):
    if "pdf_buffer" not in st.session_state:
        st.sidebar.warning("⚠️ Please generate the PDF report first.")
    elif not recipient_email:
        st.sidebar.warning("⚠️ Please enter a recipient email.")
    else:
        try:
            send_ecommerce_email(
                pdf_buffer=st.session_state["pdf_buffer"],
                recipient_email=recipient_email,
                kbi=kbi
            )
            st.sidebar.success("Email sent successfully ✅")
        except Exception as e:
            st.sidebar.error(f"Failed to send email: {e}")










