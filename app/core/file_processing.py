import pandas as pd
import re
from io import StringIO, BytesIO
from typing import Optional, Union

def get_forecast_date_from_content(file_content_stream: Union[StringIO, BytesIO]) -> Optional[pd.Timestamp]:
    """
    Extracts the forecast date from the first few lines of a file-like stream.
    Assumes the date is in a line starting with 'CENTRAL1' followed by mm/dd/yy.
    """
    try:
        # Ensure we are at the beginning of the stream
        file_content_stream.seek(0)
        # Read a few lines to find the date, assuming it's near the top
        # If it's BytesIO, we need to decode to string line by line
        is_bytes_stream = isinstance(file_content_stream, BytesIO)
        
        # Limit the number of lines to check to avoid reading entire large files
        lines_to_check = 10 
        lines_checked = 0

        for i in range(lines_to_check):
            try:
                if is_bytes_stream:
                    line_bytes = file_content_stream.readline()
                    if not line_bytes: # End of stream
                        break
                    # Try decoding with utf-8, fallback to latin-1 for file content
                    try:
                        line = line_bytes.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        line = line_bytes.decode('latin-1').strip()
                else: # StringIO
                    line_str = file_content_stream.readline()
                    if not line_str: # End of stream
                        break
                    line = line_str.strip()
            except Exception as read_err:
                print(f"⚠️ Error reading line {i+1} from stream for forecast date: {read_err}")
                break # Stop if we can't read lines

            match = re.match(r'^CENTRAL1\s+(\d{2}/\d{2}/\d{2})', line)
            if match:
                return pd.to_datetime(match.group(1), format="%m/%d/%y", errors='coerce')
            lines_checked += 1
        
        # print(f"Checked {lines_checked} lines, forecast date not found in CENTRAL1 header.") # For debugging
    except Exception as e:
        print(f"⚠️ Error processing stream for forecast date: {e}")
    return None

def parse_summary_file_from_content(file_content_stream: Union[StringIO, BytesIO]) -> pd.DataFrame:
    """
    Parses a fixed-width summary section from a .041 or .txt file-like stream 
    and returns a cleaned DataFrame with reporting dates and cash balances.
    Signs are flipped according to business rules.
    Output columns: "Previous Balance", "Opening Balance", "Net Activity", "Closing Balance", "Reporting Date"
    """
    try:
        file_content_stream.seek(0)
        content_bytes = b''
        if isinstance(file_content_stream, BytesIO):
            content_bytes = file_content_stream.read()
        elif isinstance(file_content_stream, StringIO):
            # If it's StringIO, it's already decoded text. Encode to bytes for consistent handling.
            # This might not be ideal if the original encoding was not utf-8 or latin-1,
            # but for .041 files, latin-1 is a common fallback.
            try:
                content_bytes = file_content_stream.read().encode('utf-8')
            except UnicodeEncodeError:
                file_content_stream.seek(0) # Reset for latin-1 attempt
                content_bytes = file_content_stream.read().encode('latin-1')
        else:
            print("❌ Unsupported stream type for parse_summary_file_from_content.")
            return pd.DataFrame()

        # Try decoding with UTF-8, then Latin-1
        try:
            content = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            content = content_bytes.decode('latin-1', errors='ignore')

        if not content.strip():
            print("⚠️ Summary file content is empty or whitespace only after decoding.")
            return pd.DataFrame()

        # Define fixed-width structure for summary lines
        # This is based on the original notebook's parse_summary_file
        widths = [6, 18, 19, 19, 19] # Charter, Previous Balance, Opening Balance, Net Activity, Closing Balance
        cols = ["Charter", "Previous Balance", "Opening Balance", "Net Activity", "Closing Balance"]
        
        # Use StringIO for pd.read_fwf
        data_io = StringIO(content)
        df = pd.read_fwf(data_io, widths=widths, header=None)
        
        if df.empty:
            print("⚠️ DataFrame is empty after pd.read_fwf for summary data.")
            return pd.DataFrame()
        
        df.columns = cols

        # Extract dates from CENTRAL1 lines within the content
        # This assumes summary data might be mixed with other data in the .041 file
        pattern = r'^CENTRAL1\s+(\d{2}/\d{2}/\d{2})'
        dates = []
        for line in content.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                dates.append(match.group(1))
        dates = sorted(list(set(dates))) # Get unique sorted dates

        if not dates:
            print("⚠️ No CENTRAL1 dates found in the content for summary data.")
            # If no dates found, it might mean the relevant section is missing or format is unexpected.
            # Depending on requirements, could return empty or raise an error.
            return pd.DataFrame()

        # Clean numeric formatting for balance columns
        balance_cols = cols[1:] # Exclude "Charter"
        
        # Filter rows that are likely to be summary data based on numeric pattern
        # This regex checks for typical currency format like 1,234.56 or -123.45
        numeric_pattern = r'^[+-]?\d{1,3}(?:,\d{3})*\.\d{2}$'
        
        # Keep only rows where all balance columns match the numeric pattern
        # This is a crucial step to isolate summary lines from other text in the file
        df_filtered = df[df[balance_cols].apply(lambda x: x.str.match(numeric_pattern, na=False)).all(axis=1)].copy()

        if df_filtered.empty:
            print("⚠️ No rows matched the expected numeric pattern for summary balance data.")
            return pd.DataFrame()

        for col in balance_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col].str.replace(',', '', regex=False), errors='coerce')

        # Flip signs according to convention
        def adjust_balance_sign(value):
            if pd.isna(value) or value == '':
                return value
            return -value if value > 0 else abs(value)

        for col in balance_cols:
            df_filtered[col] = df_filtered[col].apply(adjust_balance_sign)

        # Clean up and remove duplicates from the filtered data
        df_processed = df_filtered.drop(columns='Charter', errors='ignore').drop_duplicates()

        # Assign Reporting Date
        # The number of processed summary rows should match the number of unique CENTRAL1 dates found.
        # If a .041 file contains multiple summary blocks for different dates, this logic might need adjustment.
        # For now, assuming one summary block per file or that dates align correctly if multiple.
        if len(df_processed) == len(dates):
            df_processed['Reporting Date'] = pd.to_datetime(dates, format='%m/%d/%y', errors='coerce')
        elif len(df_processed) > 0 and len(dates) > 0:
            # If there's a mismatch but we have some data and some dates, 
            # this might indicate a file with multiple summary sections or an issue.
            # A common case for .041 might be one summary block per file date.
            # If there is only one summary block (one row in df_processed after filtering)
            # and one date, assign it.
            if len(df_processed) == 1 and len(dates) == 1:
                df_processed['Reporting Date'] = pd.to_datetime(dates[0], format='%m/%d/%y', errors='coerce')
            else:
                print(f"⚠️ Mismatch in processed summary row count ({len(df_processed)}) vs unique CENTRAL1 dates found ({len(dates)}). \
                      Cannot reliably assign Reporting Date. Check file structure or parsing logic for multi-date summaries.")
                # Fallback or error handling: Could try to use the latest date, or return empty.
                # For now, returning empty as date assignment is ambiguous.
                return pd.DataFrame()
        else:
            print("⚠️ No summary data rows or no CENTRAL1 dates to assign. Cannot proceed.")
            return pd.DataFrame()
        
        # Drop rows where Reporting Date could not be parsed
        df_processed.dropna(subset=['Reporting Date'], inplace=True)

        if df_processed.empty:
            print("⚠️ DataFrame is empty after assigning and dropping NaT Reporting Dates for summary.")
            return pd.DataFrame()

        # Ensure all required columns are present before returning
        final_cols = ["Previous Balance", "Opening Balance", "Net Activity", "Closing Balance", "Reporting Date"]
        df_final = df_processed[final_cols]
        
        print(f"✅ Successfully parsed summary data from content. Shape: {df_final.shape}")
        return df_final

    except Exception as e:
        import traceback
        print(f"❌ Error parsing fixed-width summary file from content: {type(e).__name__} - {str(e)}")
        traceback.print_exc() # Print full traceback for debugging
        return pd.DataFrame()

def parse_transaction_file_from_content(file_content_stream: Union[StringIO, BytesIO]) -> Optional[pd.DataFrame]:
    """
    Parses a fixed-width transaction file from a file-like stream and returns a DataFrame.
    Similar to the original parse_transaction_file but reads from a stream.
    """
    try:
        file_content_stream.seek(0)
        if isinstance(file_content_stream, BytesIO):
            try:
                content = file_content_stream.read().decode('utf-8')
            except UnicodeDecodeError:
                file_content_stream.seek(0)
                content = file_content_stream.read().decode('latin-1')
        else: # StringIO
            content = file_content_stream.read()

        if not content.strip():
            print("⚠️ Transaction file content is empty or whitespace only.")
            return None

        widths = [8, 9, 3, 18, 18, 20, 20]  # CH/BR, EFF DATE, gap, DEBIT_1, DEBIT_2, CREDIT_1, CREDIT_2
        cols = ['CH/BR', 'EFF DATE', 'GAP', 'DEBIT_1', 'DEBIT_2', 'CREDIT_1', 'CREDIT_2']
        
        data_io = StringIO(content)
        df = pd.read_fwf(data_io, widths=widths, header=None)
        if df.empty:
            print("⚠️ DataFrame is empty after pd.read_fwf for transaction file.")
            return None
        df.columns = cols

        df['Reporting Date'] = pd.NaT
        last_reporting_date = None
        for index, row in df.iterrows():
            if row['CH/BR'] == 'CENTRAL1':
                eff_date_str = str(row['EFF DATE']).strip()
                parsed_date = pd.to_datetime(eff_date_str, format='%m/%d/%y', errors='coerce')
                if not pd.isna(parsed_date):
                    last_reporting_date = parsed_date
                    df.at[index, 'Reporting Date'] = parsed_date
            
            eff_date_str = str(row['EFF DATE']).strip()
            parsed_eff_date = pd.to_datetime(eff_date_str, format='%m/%d/%y', errors='coerce')
            if not pd.isna(parsed_eff_date):
                if last_reporting_date is None:
                    last_reporting_date = parsed_eff_date
                    df.at[index, 'Reporting Date'] = parsed_eff_date
                elif parsed_eff_date <= last_reporting_date:
                    df.at[index, 'Reporting Date'] = last_reporting_date
                else:
                    last_reporting_date = parsed_eff_date
                    df.at[index, 'Reporting Date'] = parsed_eff_date
            else:
                if last_reporting_date is not None:
                    df.at[index, 'Reporting Date'] = last_reporting_date
        
        # df['Reporting Date'] = df['Reporting Date'].dt.strftime('%m/%d/%y') # Keep as datetime for now
        if 'Reporting Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Reporting Date']):
             df['Reporting Date'] = pd.to_datetime(df['Reporting Date']).dt.normalize() # Normalize to date part
        else:
            print("⚠️ Reporting Date column missing or not datetime after initial processing.")
            # Attempt conversion if it's string
            try:
                df['Reporting Date'] = pd.to_datetime(df['Reporting Date'], errors='coerce').dt.normalize()
                if df['Reporting Date'].isnull().all():
                    print("⚠️ All Reporting Dates are NaT after attempted conversion.")
                    return None # Or handle as critical error
            except Exception as date_conv_err:
                print(f"⚠️ Error converting Reporting Date to datetime: {date_conv_err}")
                return None # Critical if dates cannot be processed

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
            else:
                if re.match(date_pattern, eff_date_str):
                    last_eff_date = eff_date_str
        
        df = df.drop(columns=['CH/BR', 'GAP'], errors='ignore')
        
        if df['Reporting Date'].isnull().all():
            print("⚠️ All Reporting Dates are still NaT before returning from parse_transaction_file_from_content. This indicates a problem with date parsing.")
            return None
            
        return df

    except Exception as e:
        print(f"❌ Error parsing transaction file from content: {type(e).__name__} - {str(e)}")
        return None

def stack_debit_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stacks DEBIT and CREDIT columns, preserving EFF DATE and Reporting Date.
    Input df is expected to have columns: ['EFF DATE', 'Reporting Date', 'DEBIT_1', 'DEBIT_2', 'CREDIT_1', 'CREDIT_2']
    """
    if df is None or df.empty:
        print("⚠️ stack_debit_credit: Input DataFrame is empty or None.")
        return pd.DataFrame() # Return empty DataFrame if input is invalid

    # Ensure required columns exist
    required_cols = ['EFF DATE', 'Reporting Date', 'DEBIT_1', 'DEBIT_2', 'CREDIT_1', 'CREDIT_2']
    if not all(col in df.columns for col in required_cols):
        print(f"⚠️ stack_debit_credit: Input DataFrame missing one or more required columns: {required_cols}")
        # Attempt to return what we have, or an empty DF if critical columns like dates are missing
        if not ('EFF DATE' in df.columns and 'Reporting Date' in df.columns):
            return pd.DataFrame()
        # Fill missing debit/credit columns with empty strings or NaN to allow processing
        for col in ['DEBIT_1', 'DEBIT_2', 'CREDIT_1', 'CREDIT_2']:
            if col not in df.columns:
                df[col] = "" # or pd.NA
    
    # Stack DEBIT columns
    debit_stack = df.melt(
        id_vars=["EFF DATE", "Reporting Date"],
        value_vars=["DEBIT_1", "DEBIT_2"],
        var_name="DEBIT_Type",
        value_name="DEBIT"
    ).drop(columns=["DEBIT_Type"])
    debit_stack["DEBIT"] = debit_stack["DEBIT"].astype(str)

    # Stack CREDIT columns
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

def split_debit_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits amount and transaction type from DEBIT and CREDIT columns.
    Adjusts signs of DEBIT and CREDIT amounts based on business rules.
    Input df is expected from stack_debit_credit with 'DEBIT' and 'CREDIT' columns.
    """
    if df is None or df.empty:
        print("⚠️ split_debit_credit: Input DataFrame is empty or None.")
        return pd.DataFrame()

    # Ensure 'DEBIT' and 'CREDIT' columns exist, even if from a partial stack
    if 'DEBIT' not in df.columns:
        df['DEBIT'] = ""
    if 'CREDIT' not in df.columns:
        df['CREDIT'] = ""

    df['DEBIT'] = df['DEBIT'].fillna('').astype(str)
    # Regex to capture amount and a 2 or 3 char/digit code
    df[['DEBIT_AMOUNT_STR', 'DEBIT_TRANSACTION_TYPE']] = df['DEBIT'].str.extract(r'(-?"?[\d,]*\.\d{2})\s*([A-Z0-9]{2,3})?$')
    df['DEBIT'] = pd.to_numeric(df['DEBIT_AMOUNT_STR'].str.replace(',', '', regex=False).str.replace('"', '', regex=False), errors='coerce')
    df.drop(columns=['DEBIT_AMOUNT_STR'], inplace=True, errors='ignore')

    df['CREDIT'] = df['CREDIT'].fillna('').astype(str)
    df[['CREDIT_AMOUNT_STR', 'CREDIT_TRANSACTION_TYPE']] = df['CREDIT'].str.extract(r'(-?"?[\d,]*\.\d{2})\s*([A-Z0-9]{2,3})?$')
    df['CREDIT'] = pd.to_numeric(df['CREDIT_AMOUNT_STR'].str.replace(',', '', regex=False).str.replace('"', '', regex=False), errors='coerce')
    df.drop(columns=['CREDIT_AMOUNT_STR'], inplace=True, errors='ignore')

    # Fill NaNs in transaction type columns that might result from non-matching regex
    df['DEBIT_TRANSACTION_TYPE'] = df['DEBIT_TRANSACTION_TYPE'].fillna('')
    df['CREDIT_TRANSACTION_TYPE'] = df['CREDIT_TRANSACTION_TYPE'].fillna('')

    # Filter rows where either DEBIT_TRANSACTION_TYPE or CREDIT_TRANSACTION_TYPE is valid (has 2 or 3 chars)
    # This was previously: pattern = r'^[A-Z]{1,2}[A-Z0-9]?$' which is too restrictive for 3 char codes
    # A simpler check for 2 or 3 character codes:
    valid_code_pattern = r'^[A-Z0-9]{2,3}$'
    mask = (
        df['DEBIT_TRANSACTION_TYPE'].str.match(valid_code_pattern) |
        df['CREDIT_TRANSACTION_TYPE'].str.match(valid_code_pattern)
    )
    # If only one side has a transaction, the other side's amount should be NaN or 0 after extraction
    # The mask ensures we keep rows that have at least one valid transaction code.
    df = df[mask].reset_index(drop=True)
    if df.empty:
        print("⚠️ split_debit_credit: DataFrame is empty after filtering for valid transaction codes.")
        return pd.DataFrame()

    def adjust_debit_sign(value):
        if pd.isna(value) or value == '': return value
        return -value if value > 0 else abs(value)

    def adjust_credit_sign(value):
        if pd.isna(value) or value == '': return value
        return abs(value) if value < 0 else value

    df['DEBIT'] = df['DEBIT'].apply(adjust_debit_sign)
    df['CREDIT'] = df['CREDIT'].apply(adjust_credit_sign)
    return df

def transpose_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transposes the DataFrame into a standard transaction format.
    Input df is expected from split_debit_credit.
    """
    if df is None or df.empty:
        print("⚠️ transpose_transactions: Input DataFrame is empty or None.")
        return pd.DataFrame()
    
    # Ensure necessary columns from split_debit_credit are present
    required_cols_transpose = ['EFF DATE', 'Reporting Date', 'DEBIT', 'DEBIT_TRANSACTION_TYPE', 'CREDIT', 'CREDIT_TRANSACTION_TYPE']
    if not all(col in df.columns for col in required_cols_transpose):
        print(f"⚠️ transpose_transactions: Input DataFrame missing one or more required columns for transposing.")
        return pd.DataFrame() # Cannot proceed without these columns

    debit_df = df[df['DEBIT_TRANSACTION_TYPE'] != ''][['EFF DATE', 'Reporting Date', 'DEBIT', 'DEBIT_TRANSACTION_TYPE']].rename(
        columns={'EFF DATE': 'Eff Date', 'DEBIT': 'Transaction Amount', 'DEBIT_TRANSACTION_TYPE': 'Transaction Code'}
    )
    if not debit_df.empty:
        debit_df['Transaction Type'] = 'Debit'

    credit_df = df[df['CREDIT_TRANSACTION_TYPE'] != ''][['EFF DATE', 'Reporting Date', 'CREDIT', 'CREDIT_TRANSACTION_TYPE']].rename(
        columns={'EFF DATE': 'Eff Date', 'CREDIT': 'Transaction Amount', 'CREDIT_TRANSACTION_TYPE': 'Transaction Code'}
    )
    if not credit_df.empty:
        credit_df['Transaction Type'] = 'Credit'
    
    # Ensure Transaction Amount from credit_df does not become negative based on original rule
    # The original rule was: "If Transaction Type is Credit, Transaction Amount becomes negative."
    # However, adjust_credit_sign already ensures credits are positive (or abs of negative reversals).
    # The final sign for the unified 'Transaction Amount' column should reflect debit (-) and credit (+)
    # Let's adjust here for the final unified column:
    if not debit_df.empty:
        # Debits are generally outflows, often represented as negative in final transaction lists.
        # adjust_debit_sign already made them negative if they were positive outflows.
        pass # Debit amounts are already correctly signed

    if not credit_df.empty:
        # Credits are inflows, should be positive. adjust_credit_sign ensures this.
        pass # Credit amounts are already correctly signed

    # If both are empty, return empty
    if debit_df.empty and credit_df.empty:
        return pd.DataFrame()
        
    combined_df = pd.concat([debit_df, credit_df], ignore_index=True)
    combined_df = combined_df.dropna(subset=['Transaction Amount', 'Transaction Code']).reset_index(drop=True)

    # Ensure date columns are properly parsed and formatted if they are not already datetime
    # The parse_transaction_file_from_content should have handled Reporting Date to datetime.
    # EFF DATE might still be string from earlier steps.
    try:
        combined_df['Eff Date'] = pd.to_datetime(combined_df['Eff Date'], format='%m/%d/%y', errors='coerce').dt.strftime('%Y-%m-%d')
    except AttributeError: # If already datetime
        combined_df['Eff Date'] = pd.to_datetime(combined_df['Eff Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    try:
        combined_df['Reporting Date'] = pd.to_datetime(combined_df['Reporting Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    except AttributeError: # If already datetime
         combined_df['Reporting Date'] = pd.to_datetime(combined_df['Reporting Date'], errors='coerce').dt.strftime('%Y-%m-%d')

    final_cols = ['Eff Date', 'Reporting Date', 'Transaction Amount', 'Transaction Type', 'Transaction Code']
    # Ensure all final columns are present before reordering
    for col in final_cols:
        if col not in combined_df.columns:
            print(f"Warning: Final column '{col}' missing before reordering in transpose_transactions.")
            combined_df[col] = pd.NA # Add as NA if missing
            
    combined_df = combined_df[final_cols]
    return combined_df

def add_transaction_category(final_df: pd.DataFrame, mapping_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'Transaction Category' to the transaction dataframe by mapping Transaction Code
    using the provided mapping_data DataFrame.

    Parameters:
    - final_df (pd.DataFrame): The main transaction dataframe, expected to have 'Transaction Code'.
    - mapping_data (pd.DataFrame): DataFrame containing the transaction code mappings,
                                   expected to have columns like 'TranCode' and 'TranCategory'.

    Returns:
    - pd.DataFrame: The input DataFrame with an added 'Transaction Category' column.
    """
    if final_df is None or final_df.empty:
        print("⚠️ add_transaction_category: Input final_df is empty or None.")
        return pd.DataFrame()
    if mapping_data is None or mapping_data.empty:
        print("⚠️ add_transaction_category: Input mapping_data is empty or None. Cannot map categories.")
        # Return df as is, but add an empty/default category column to maintain schema if needed
        if 'Transaction Category' not in final_df.columns:
             final_df['Transaction Category'] = 'Other' # Default if no mapping applied
        return final_df

    # Ensure required columns exist in mapping_data
    if not all(col in mapping_data.columns for col in ['TranCode', 'TranCategory']):
        print("⚠️ add_transaction_category: mapping_data is missing 'TranCode' or 'TranCategory' columns.")
        if 'Transaction Category' not in final_df.columns:
             final_df['Transaction Category'] = 'Other'
        return final_df

    # Create mapping dictionary from the mapping_data DataFrame
    # Ensure TranCode in mapping_data is string for consistent mapping
    tran_code_to_category = dict(zip(mapping_data['TranCode'].astype(str), mapping_data['TranCategory'])) 

    # Map 'Transaction Category' into final_df
    # Ensure 'Transaction Code' in final_df is also string for consistent mapping
    if 'Transaction Code' in final_df.columns:
        final_df['Transaction Category'] = final_df['Transaction Code'].astype(str).map(tran_code_to_category).fillna('Other')
    else:
        print("⚠️ add_transaction_category: 'Transaction Code' column not found in final_df.")
        final_df['Transaction Category'] = 'Other' # Default if no code to map from
    
    # Sort by Reporting Date (assuming 'Reporting Date' exists and is datetime or sortable string)
    if 'Reporting Date' in final_df.columns:
        try:
            # Attempt to convert to datetime if not already, to ensure proper sorting
            final_df['Reporting Date'] = pd.to_datetime(final_df['Reporting Date'], errors='coerce')
            final_df = final_df.sort_values(by='Reporting Date', ascending=True)
        except Exception as sort_err:
            print(f"⚠️ add_transaction_category: Could not sort by Reporting Date. Error: {sort_err}")
    else:
        print("⚠️ add_transaction_category: 'Reporting Date' column not found for sorting.")

    # print(f"✅ Added 'Transaction Category' column successfully.") # Optional success message
    return final_df

def process_transaction_data_from_content(
    transaction_file_content_stream: Union[StringIO, BytesIO],
    mapping_data_df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """
    Processes raw transaction data from a file stream through all parsing steps.

    Parameters:
    - transaction_file_content_stream: File-like stream (StringIO or BytesIO) of the transaction data.
    - mapping_data_df (Optional[pd.DataFrame]): DataFrame with transaction code mappings 
                                                 (columns 'TranCode', 'TranCategory'). 
                                                 If None, categories will not be added or will default to 'Other'.

    Returns:
    - pd.DataFrame: Cleaned and processed transaction data, or None if a critical error occurs.
    """
    print("🚀 Starting transaction data processing...")
    df_parsed = parse_transaction_file_from_content(transaction_file_content_stream)
    if df_parsed is None or df_parsed.empty:
        print("❌ Transaction processing halted: Initial parsing failed or returned empty.")
        return None
    print("  ✅ Step 1/5: Initial parsing complete.")

    df_stacked = stack_debit_credit(df_parsed)
    if df_stacked.empty:
        print("❌ Transaction processing halted: Stacking debit/credit failed or returned empty.")
        # df_parsed might still be useful if stacking is optional or has issues
        # For now, consider it a failure in the chain if an intermediate step returns empty
        return None 
    print("  ✅ Step 2/5: Debit/credit stacking complete.")

    df_split = split_debit_credit(df_stacked)
    if df_split.empty:
        print("❌ Transaction processing halted: Splitting debit/credit failed or returned empty.")
        return None
    print("  ✅ Step 3/5: Debit/credit splitting complete.")

    df_transposed = transpose_transactions(df_split)
    if df_transposed.empty:
        print("❌ Transaction processing halted: Transposing transactions failed or returned empty.")
        return None
    print("  ✅ Step 4/5: Transaction transposing complete.")

    if mapping_data_df is not None and not mapping_data_df.empty:
        df_categorized = add_transaction_category(df_transposed, mapping_data_df)
        print("  ✅ Step 5/5: Transaction categorization applied.")
    else:
        print("  ⚠️ Step 5/5: Mapping data not provided or empty. Skipping transaction categorization or applying default.")
        # Ensure 'Transaction Category' column exists even if not mapped, for consistent schema
        if 'Transaction Category' not in df_transposed.columns:
            df_transposed['Transaction Category'] = 'Other'
        df_categorized = df_transposed # Use the transposed df if no mapping applied
    
    print("🏁 Transaction data processing finished.")
    return df_categorized

def parse_liquidity_template_from_content(file_content_stream: BytesIO) -> Optional[pd.DataFrame]:
    """
    Parses the liquidity template Excel file (Cash Flow Workbook Template) from a BytesIO stream.
    Calculates liquidity ratios and extracts a reporting date.

    Returns:
    - pd.DataFrame with columns ["Ratio Name", "Ratio (%)", "Reporting Date"], or None on error.
    """
    try:
        file_content_stream.seek(0)
        df = pd.read_excel(file_content_stream, engine='openpyxl') # Specify engine for .xlsx

        if df.empty:
            print("⚠️ Liquidity template is empty after reading Excel.")
            return None

        # Normalize column names (remove leading/trailing spaces)
        df.columns = [str(col).strip() for col in df.columns]
        
        # Ensure 'Code' and 'Value' columns exist, attempt to find them if names vary slightly
        code_col_name = next((col for col in df.columns if 'code' in col.lower()), None)
        value_col_name = next((col for col in df.columns if 'value' in col.lower()), None)

        if not code_col_name or not value_col_name:
            print(f"⚠️ Liquidity template missing required 'Code' or 'Value' columns. Found: {df.columns}")
            return None
            
        df.rename(columns={code_col_name: 'Code', value_col_name: 'Value'}, inplace=True)

        df["Code"] = df["Code"].astype(str).str.strip()
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0) # Fill non-numeric with 0

        # Extract the reporting date (look for a column named similar to 'Date')
        possible_date_cols = [col for col in df.columns if 'date' in col.lower()]
        reporting_date: Optional[pd.Timestamp] = None
        if possible_date_cols:
            date_col_name = possible_date_cols[0]
            # Try to get the first valid date from that column
            valid_dates = pd.to_datetime(df[date_col_name], errors='coerce').dropna()
            if not valid_dates.empty:
                reporting_date = valid_dates.iloc[0]
            else:
                print(f"⚠️ No valid dates found in column '{date_col_name}' for liquidity template.")
        else:
            print("⚠️ No 'Date' column found in liquidity template to determine reporting date.")
        
        # If reporting_date is still None, we might need a default or raise error
        # For now, proceed, it will be NaT in the output if not found

        statutory_liquidity_codes = ['1010', '1040', '1060', '1078', '1079']
        deposit_and_debt_codes = ['2180', '2050', '2255', '2295'] # Original had 2050 (Borrowings) and 2180 (Member Deposits)
        core_cash_codes = ['1100'] # e.g. Cash and Equivalents
        borrowings_codes = ['2050']
        member_deposits_codes = ['2180']

        liquidity_available = df[df["Code"].isin(statutory_liquidity_codes)]["Value"].sum()
        deposits_and_debt_total = df[df["Code"].isin(deposit_and_debt_codes)]["Value"].sum()
        total_cash_liquid = df[df["Code"].isin(core_cash_codes)]["Value"].sum()
        borrowings_total = df[df["Code"].isin(borrowings_codes)]["Value"].sum()
        member_deposits_total = df[df["Code"].isin(member_deposits_codes)]["Value"].sum()

        statutory_ratio = (liquidity_available / deposits_and_debt_total) if deposits_and_debt_total else 0
        # Core Liquidity: (Core Cash - Borrowings) / Member Deposits
        core_ratio = ((total_cash_liquid - borrowings_total) / member_deposits_total) if member_deposits_total else 0
        # Total Liquidity: Core Cash / Member Deposits (assuming this is the intent)
        total_ratio = (total_cash_liquid / member_deposits_total) if member_deposits_total else 0

        result_df = pd.DataFrame({
            "Ratio Name": [
                "Statutory Liquidity Ratio",
                "Core Liquidity Ratio",
                "Total Liquidity Ratio"
            ],
            "Ratio (%)": [
                round(statutory_ratio * 100, 2),
                round(core_ratio * 100, 2),
                round(total_ratio * 100, 2)
            ],
            "Reporting Date": [reporting_date] * 3 # Will be NaT if not found
        })
        print("✅ Liquidity Ratios parsed successfully.")
        return result_df

    except Exception as e:
        print(f"❌ Error parsing liquidity template from content: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def parse_usd_exposure_template_from_content(file_content_stream: BytesIO) -> Optional[pd.DataFrame]:
    """
    Parses the USD Exposure template Excel file from a BytesIO stream.
    Calculates USD Exposure = Total Assets + Total Liabilities + Total Capital.
    Extracts the reporting date from the 'Total Asset' row.

    Args:
        file_content_stream (BytesIO): The content of the Excel file as a BytesIO stream.

    Returns:
        pd.DataFrame with columns ["usd_exposure_value", "Reporting Date"], or None on error.
    """
    try:
        file_content_stream.seek(0) # Ensure stream is at the beginning
        df = pd.read_excel(file_content_stream, engine='openpyxl') # Specify engine for .xlsx

        if df.empty:
            print("⚠️ USD Exposure template is empty after reading Excel.")
            return None

        # Normalize column names (remove leading/trailing spaces)
        df.columns = [str(col).strip() for col in df.columns]
        
        # Ensure 'Description', 'Value', 'Date' columns exist by trying to find them
        description_col_name = next((col for col in df.columns if 'description' in col.lower()), None)
        value_col_name = next((col for col in df.columns if 'value' in col.lower()), None)
        date_col_name = next((col for col in df.columns if 'date' in col.lower()), None)

        if not description_col_name or not value_col_name: # Date column can be optional for calculation but needed for date extraction
            print(f"⚠️ USD Exposure template missing required 'Description' or 'Value' columns. Found: {df.columns}")
            return None
        
        # Rename to standard names used in logic
        rename_map = {description_col_name: 'Description', value_col_name: 'Value'}
        if date_col_name:
            rename_map[date_col_name] = 'Date'
        df.rename(columns=rename_map, inplace=True)

        df["Description"] = df["Description"].astype(str).str.strip().str.lower()
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0)

        # Extract values
        total_asset = df.loc[df["Description"] == "total asset", "Value"].sum()
        total_liabilities = df.loc[df["Description"] == "total liabilities", "Value"].sum()
        total_capital = df.loc[df["Description"] == "total capital", "Value"].sum()

        # Calculate USD Exposure
        usd_exposure = round(total_asset + total_liabilities + total_capital, 2)

        # Extract the date from 'total asset' row if 'Date' column exists
        exposure_date: Optional[pd.Timestamp] = None
        if 'Date' in df.columns:
            date_series = df.loc[df["Description"] == "total asset", "Date"]
            if not date_series.empty:
                # Try to convert the first non-null date value
                valid_dates = pd.to_datetime(date_series, errors='coerce').dropna()
                if not valid_dates.empty:
                    exposure_date = valid_dates.iloc[0]
                else:
                    print(f"⚠️ No valid date found in 'Date' column for 'total asset' row in USD Exposure template.")
            else:
                print(f"⚠️ 'total asset' row found, but 'Date' column is empty or NaT in USD Exposure template.")
        else:
            print("⚠️ 'Date' column not found in USD Exposure template. Cannot extract exposure date.")


        return pd.DataFrame({
            "usd_exposure_value": [usd_exposure],
            "Reporting Date": [exposure_date] # This will be NaT if not found/extracted
        })

    except Exception as e:
        print(f"❌ Error parsing USD Exposure template from content: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

# Helper function to check for Excel file signature (optional, but good practice)
# This might be useful for the router to decide if BytesIO should be used for Excel
def is_excel_file_signature(file_content_bytes: bytes) -> bool:
    """
    Checks if the initial bytes of the file content match common Excel signatures.
    PKZIP for .xlsx (Office Open XML), OLECF for .xls (BIFF).
    """
    if file_content_bytes is None or len(file_content_bytes) < 8:
        return False
    # .xlsx (PKZIP, starts with PK\\x03\\x04)
    if file_content_bytes.startswith(b'PK\\x03\\x04'):
        return True
    # .xls (OLECF, starts with \\xd0\\xcf\\x11\\xe0\\xa1\\xb1\\x1a\\xe1)
    if file_content_bytes.startswith(b'\\xd0\\xcf\\x11\\xe0\\xa1\\xb1\\x1a\\xe1'): # d0cf11e0a1b11ae1
        return True
    return False

# Placeholder for actual forecast model execution
# def run_forecast_model(db_client: Client, user_id: uuid.UUID, forecast_parameters: Dict) -> Dict:
# print("Forecast model execution started with parameters:", forecast_parameters)
#     # 1. Fetch necessary data from Supabase tables based on user_id and parameters
#     #    (e.g., transaction_data, summary_data for the relevant period and currency)
#     # 2. Preprocess data for the model (similar to your main.py logic)
#     # 3. Load/train forecasting model (e.g., RandomForestRegressor)
#     # 4. Generate forecast for the specified horizon
#     # 5. Format results (e.g., as a list of dicts or a DataFrame to be saved as CSV)
#     # 6. Store forecast results (e.g., as a CSV in Supabase Storage, and metadata in a forecast_results table)
#     # Example result:
#     # forecast_result_path = f"{user_id}/forecasts/{datetime.now().strftime('%Y%m%d%H%M%S')}_forecast.csv"
#     # Store the actual forecast data (e.g., a DataFrame converted to CSV bytes) to this path in Supabase Storage
# return {"status": "completed", "result_path": "path/to/forecast_output.csv", "message": "Forecast generated successfully."} 