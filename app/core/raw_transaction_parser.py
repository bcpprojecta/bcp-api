import re
from io import StringIO, BytesIO
from datetime import datetime
import pandas as pd
import numpy as np
from supabase import Client # For type hinting

# --- Helper function from original main.py to get reporting date from file header ---
# This might be useful if the reporting date needs to be cross-checked or extracted
# directly from content, though the main parsing logic in main.py also derives it.
def get_reporting_date_from_header(file_content_stream: BytesIO) -> pd.Timestamp | None:
    try:
        file_content_stream.seek(0) # Ensure stream is at the beginning
        # Read a few lines to find the header, assuming latin-1 encoding for these specific files
        # Limit reading to avoid loading entire large file into memory just for header
        header_content = file_content_stream.read(2048).decode('latin-1', errors='ignore')
        file_content_stream.seek(0) # Reset stream for further processing by other functions

        for line in header_content.splitlines():
            match = re.match(r'^CENTRAL1\s+(\d{2}/\d{2}/\d{2})', line.strip())
            if match:
                return pd.to_datetime(match.group(1), format="%m/%d/%y", errors='coerce')
    except Exception as e:
        print(f"⚠️ Error reading reporting date from header: {e}")
    return None

# --- Core parsing functions adapted from BCP DATA/main.py ---

def parse_transaction_content(file_content_stream: BytesIO) -> pd.DataFrame | None:
    """Parses the content of a .041 or .txt transaction file (fixed-width) from a BytesIO stream.
    
    Returns a pandas DataFrame with raw transaction data.
    """
    try:
        file_content_stream.seek(0)
        # Try reading with UTF-8 first, fall back to latin-1 if it fails
        try:
            content = file_content_stream.read().decode('utf-8')
        except UnicodeDecodeError:
            file_content_stream.seek(0) # Reset stream
            content = file_content_stream.read().decode('latin-1', errors='ignore')
        file_content_stream.seek(0) # Reset for any further use

        # Define column widths
        widths = [8, 9, 3, 18, 18, 20, 20]  # CH/BR, EFF DATE, gap, DEBIT_1, DEBIT_2, CREDIT_1, CREDIT_2
        df = pd.read_fwf(StringIO(content), widths=widths, header=None)
        df.columns = ['CH/BR', 'EFF DATE', 'GAP', 'DEBIT_1', 'DEBIT_2', 'CREDIT_1', 'CREDIT_2']
        
        df['Reporting Date'] = pd.NaT
        last_reporting_date = None

        for index, row in df.iterrows():
            if row['CH/BR'] == 'CENTRAL1':
                eff_date_str = str(row['EFF DATE']).strip()
                parsed_date = pd.to_datetime(eff_date_str, format='%m/%d/%y', errors='coerce')
                if not pd.isna(parsed_date):
                    last_reporting_date = parsed_date
                    df.at[index, 'Reporting Date'] = parsed_date
            else: # For non-header rows, apply reporting date logic based on EFF DATE
                eff_date_str = str(row['EFF DATE']).strip()
                parsed_eff_date = pd.to_datetime(eff_date_str, format='%m/%d/%y', errors='coerce')
                if not pd.isna(parsed_eff_date):
                    if last_reporting_date is None: # Should ideally be set by a CENTRAL1 line first
                        last_reporting_date = parsed_eff_date
                    # If current EFF DATE is later than last_reporting_date, update last_reporting_date
                    # This logic might need refinement based on exact file specs for multi-date files
                    if parsed_eff_date > last_reporting_date:
                         last_reporting_date = parsed_eff_date
                    df.at[index, 'Reporting Date'] = last_reporting_date

                elif last_reporting_date is not None: # Carry forward if EFF DATE is invalid/missing
                    df.at[index, 'Reporting Date'] = last_reporting_date
        
        # Ensure Reporting Date is filled if possible, drop rows where it's still NaT
        df.dropna(subset=['Reporting Date'], inplace=True)
        if df.empty:
            return None

        df['Reporting Date'] = pd.to_datetime(df['Reporting Date'])


        date_pattern = r'^\d{2}/\d{2}/\d{2}$'
        df['EFF DATE'] = df['EFF DATE'].astype(str).replace('nan', '')
        mask = (df['EFF DATE'].str.match(date_pattern)) | (df['EFF DATE'] == '')
        df = df[mask].reset_index(drop=True)

        last_eff_date = None
        for index, row in df.iterrows():
            eff_date_str = row['EFF DATE'].strip()
            if eff_date_str == '':
                if last_eff_date is not None:
                    df.at[index, 'EFF DATE'] = last_eff_date
            elif re.match(date_pattern, eff_date_str):
                last_eff_date = eff_date_str
        
        df = df.drop(columns=['CH/BR', 'GAP'])
        return df

    except Exception as e:
        print(f"❌ Error in parse_transaction_content: {type(e).__name__} - {str(e)}")
        return None

def stack_debit_credit(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    try:
        debit_stack = df.melt(
            id_vars=["EFF DATE", "Reporting Date"], 
            value_vars=["DEBIT_1", "DEBIT_2"], 
            var_name="DEBIT_Type", 
            value_name="DEBIT"
        ).drop(columns=["DEBIT_Type"])
        debit_stack["DEBIT"] = debit_stack["DEBIT"].astype(str)

        credit_stack = df.melt(
            id_vars=["EFF DATE", "Reporting Date"], 
            value_vars=["CREDIT_1", "CREDIT_2"], 
            var_name="CREDIT_Type", 
            value_name="CREDIT"
        ).drop(columns=["CREDIT_Type"])
        credit_stack["CREDIT"] = credit_stack["CREDIT"].astype(str)

        stacked_df = pd.concat([debit_stack, credit_stack["CREDIT"]], axis=1)
        stacked_df = stacked_df[
            ~((stacked_df["DEBIT"] == "") & (stacked_df["CREDIT"] == "")) &
            ~((stacked_df["DEBIT"].str.lower() == "nan") & (stacked_df["CREDIT"].str.lower() == "nan"))
        ].reset_index(drop=True)
        return stacked_df
    except Exception as e:
        print(f"❌ Error in stack_debit_credit: {type(e).__name__} - {str(e)}")
        return None

def split_debit_credit(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    try:
        df['DEBIT'] = df['DEBIT'].fillna('').astype(str)
        df[['DEBIT_Amount_Str', 'DEBIT_TRANSACTION_TYPE']] = df['DEBIT'].str.extract(r'(-?"?[\d,]*\.?\d{0,2})\s*([A-Z0-9]{2,3})?')
        df['DEBIT'] = pd.to_numeric(df['DEBIT_Amount_Str'].str.replace(',', ''), errors='coerce')
        
        df['CREDIT'] = df['CREDIT'].fillna('').astype(str)
        df[['CREDIT_Amount_Str', 'CREDIT_TRANSACTION_TYPE']] = df['CREDIT'].str.extract(r'(-?"?[\d,]*\.?\d{0,2})\s*([A-Z0-9]{2,3})?')
        df['CREDIT'] = pd.to_numeric(df['CREDIT_Amount_Str'].str.replace(',', ''), errors='coerce')

        df.drop(columns=['DEBIT_Amount_Str', 'CREDIT_Amount_Str'], inplace=True)
        
        pattern = r'^[A-Z]{1,2}[A-Z0-9]?$'
        df['DEBIT_TRANSACTION_TYPE'] = df['DEBIT_TRANSACTION_TYPE'].fillna('')
        df['CREDIT_TRANSACTION_TYPE'] = df['CREDIT_TRANSACTION_TYPE'].fillna('')
        
        mask = (
            df['DEBIT_TRANSACTION_TYPE'].str.match(pattern) |
            df['CREDIT_TRANSACTION_TYPE'].str.match(pattern)
        )
        df = df[mask].reset_index(drop=True)

        def adjust_debit_sign(value):
            if pd.isna(value) or value == '': return value
            return -value if value > 0 else abs(value)

        def adjust_credit_sign(value):
            if pd.isna(value) or value == '': return value
            return abs(value) if value < 0 else value

        df['DEBIT'] = df['DEBIT'].apply(adjust_debit_sign)
        df['CREDIT'] = df['CREDIT'].apply(adjust_credit_sign)
        return df
    except Exception as e:
        print(f"❌ Error in split_debit_credit: {type(e).__name__} - {str(e)}")
        return None

def transpose_transactions(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    try:
        debit_df = df[df['DEBIT_TRANSACTION_TYPE'] != ''][['EFF DATE', 'Reporting Date', 'DEBIT', 'DEBIT_TRANSACTION_TYPE']].rename(
            columns={'EFF DATE': 'Eff Date', 'DEBIT': 'Transaction Amount', 'DEBIT_TRANSACTION_TYPE': 'Transaction Code'}
        )
        debit_df['Transaction Type'] = 'Debit'
        
        credit_df = df[df['CREDIT_TRANSACTION_TYPE'] != ''][['EFF DATE', 'Reporting Date', 'CREDIT', 'CREDIT_TRANSACTION_TYPE']].rename(
            columns={'EFF DATE': 'Eff Date', 'CREDIT': 'Transaction Amount', 'CREDIT_TRANSACTION_TYPE': 'Transaction Code'}
        )
        credit_df['Transaction Type'] = 'Credit'
        
        combined_df = pd.concat([debit_df, credit_df], ignore_index=True)
        combined_df = combined_df.dropna(subset=['Transaction Amount', 'Transaction Code']).reset_index(drop=True)
        
        # Convert to datetime first, then format to string to avoid issues with NaT
        combined_df['Eff Date'] = pd.to_datetime(combined_df['Eff Date'], format='%m/%d/%y', errors='coerce')
        combined_df['Reporting Date'] = pd.to_datetime(combined_df['Reporting Date'], errors='coerce') # Already datetime from parse_transaction_content

        # Drop rows where dates could not be parsed
        combined_df.dropna(subset=['Eff Date', 'Reporting Date'], inplace=True)

        # Now format to string YYYY-MM-DD for database
        combined_df['Eff Date'] = combined_df['Eff Date'].dt.strftime('%Y-%m-%d')
        combined_df['Reporting Date'] = combined_df['Reporting Date'].dt.strftime('%Y-%m-%d')
        
        return combined_df[['Eff Date', 'Reporting Date', 'Transaction Amount', 'Transaction Type', 'Transaction Code']]
    except Exception as e:
        print(f"❌ Error in transpose_transactions: {type(e).__name__} - {str(e)}")
        return None

def add_transaction_category_from_db(df: pd.DataFrame, db_client: Client) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    try:
        # Select TranCode, TranDescription, and TranCategory
        mapping_response = db_client.table("transaction_code_mappings").select("TranCode, TranDescription, TranCategory").execute()
        
        if not mapping_response.data:
            print("⚠️ No mapping data found in transaction_code_mappings. 'Transaction Category' will be 'Other'.")
            df['transaction_category'] = 'Other'
            return df.sort_values(by='Reporting Date', ascending=True)

        mapping_df = pd.DataFrame(mapping_response.data)
        tran_code_to_category = dict(zip(mapping_df['TranCode'], mapping_df['TranCategory']))
        
        df['transaction_category'] = df['Transaction Code'].map(tran_code_to_category).fillna('Other')
        df = df.sort_values(by='Reporting Date', ascending=True)
        return df
    except Exception as e:
        print(f"❌ Error in add_transaction_category_from_db: {type(e).__name__} - {str(e)}. Assigning 'Other' to categories.")
        if df is not None:
            df['transaction_category'] = 'Other'
            return df.sort_values(by='Reporting Date', ascending=True)
        return None

def process_041_content(file_content_stream: BytesIO, db_client: Client, currency_code: str) -> pd.DataFrame | None:
    """Main processing pipeline for a .041 transaction file stream.
    
    Takes BytesIO stream of the file, Supabase client, and currency_code (e.g., 'CAD', 'USD').
    Returns a cleaned DataFrame ready for DB insertion 
    (columns: Eff Date, Reporting Date, Transaction Amount, Transaction Type, Transaction Code, transaction_category, currency).
    """
    print(f"🚀 Starting .041 file processing for currency: {currency_code}...")
    
    df_parsed = parse_transaction_content(file_content_stream)
    if df_parsed is None or df_parsed.empty:
        print("❌ Parsing transaction content failed or returned empty.")
        return None
    print(f"✅ Parsed content. Shape: {df_parsed.shape}")

    df_stacked = stack_debit_credit(df_parsed)
    if df_stacked is None or df_stacked.empty:
        print("❌ Stacking debit/credit failed or returned empty.")
        return None
    print(f"✅ Stacked debit/credit. Shape: {df_stacked.shape}")

    df_split = split_debit_credit(df_stacked)
    if df_split is None or df_split.empty:
        print("❌ Splitting debit/credit failed or returned empty.")
        return None
    print(f"✅ Split debit/credit. Shape: {df_split.shape}")
    
    df_transposed = transpose_transactions(df_split)
    if df_transposed is None or df_transposed.empty:
        print("❌ Transposing transactions failed or returned empty.")
        return None
    print(f"✅ Transposed transactions. Shape: {df_transposed.shape}")

    df_categorized = add_transaction_category_from_db(df_transposed, db_client)
    if df_categorized is None or df_categorized.empty:
        print("❌ Adding transaction category failed or returned empty.")
        if df_transposed is not None: # Return transposed even if categorization fails
             df_transposed['transaction_category'] = 'Other'
             if currency_code:
                 df_transposed['currency'] = currency_code.upper()
             else:
                 print("⚠️ Currency code not provided to process_041_content, 'currency' column will be missing.")
             return df_transposed.sort_values(by='Reporting Date', ascending=True)
        return None
    print(f"✅ Added transaction categories. Final shape: {df_categorized.shape}")
    
    # Add currency column
    if currency_code:
        df_categorized['currency'] = currency_code.upper()
        print(f"✅ Added 'currency' column with value: {currency_code.upper()}. Shape after adding currency: {df_categorized.shape}")
    else:
        print("⚠️ Currency code not provided to process_041_content, 'currency' column will be missing from the final DataFrame.")

    # Final check for expected columns for Output table
    expected_cols = ['Eff Date', 'Reporting Date', 'Transaction Amount', 'Transaction Type', 'Transaction Code', 'transaction_category', 'currency']
    missing_cols = [col for col in expected_cols if col not in df_categorized.columns]
    if missing_cols:
        print(f"⚠️ WARNING: Final DataFrame from process_041_content is missing expected columns for 'Output' table: {missing_cols}")
    else:
        print(f"✅ Final DataFrame columns for 'Output' table are present: {df_categorized.columns.tolist()}")

    return df_categorized 