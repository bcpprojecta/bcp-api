import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List
import uuid # For type hinting user_id, job_id
from supabase import Client # For type hinting db client
from sklearn.ensemble import RandomForestRegressor # For forecasting model
import traceback # For logging errors

# --- Feature Engineering Helper ---
def add_features(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag and calendar-based features to the time series data.
    Assumes datetime index is set. Operates purely on the DataFrame.
    """
    # Ensure index is datetime
    if not isinstance(df_ml.index, pd.DatetimeIndex):
        # print("⚠️ Warning: Index is not DatetimeIndex in add_features. Attempting conversion.") # MODIFICATION: Commented out
        try:
            df_ml.index = pd.to_datetime(df_ml.index)
        except Exception as e:
            # print(f"❌ Error converting index to DatetimeIndex: {e}") # MODIFICATION: Commented out
            # Depending on robustness needed, either raise error or return df as is
            raise ValueError("Index could not be converted to DatetimeIndex.") from e

    df_ml['dayofweek'] = df_ml.index.dayofweek
    df_ml['month'] = df_ml.index.month
    df_ml['dayofmonth'] = df_ml.index.day
    df_ml['is_weekend'] = (df_ml['dayofweek'] >= 5).astype(int)
    df_ml['is_month_end'] = df_ml.index.is_month_end.astype(int)
    # Add features from notebook
    df_ml['days_from_month_start'] = df_ml.index.day - 1
    df_ml['days_to_month_end'] = df_ml.index.days_in_month - df_ml.index.day
    # Simple lag features (more complex features could be added)
    df_ml['lag1'] = df_ml['Amount'].shift(1)
    # Add more lags or rolling features if needed - Temporarily commented out for debugging
    # df_ml['lag7'] = df_ml['Amount'].shift(7)
    # df_ml['rolling_mean_7'] = df_ml['Amount'].rolling(window=7, min_periods=1).mean().shift(1) # Use min_periods=1 for rolling

    # Drop rows with NaN created by shifts/rolling features
    # print(f"    DEBUG add_features: Input df_ml shape: {df_ml.shape}, NaN count before drop: {df_ml.isnull().sum().sum()}") # MODIFICATION: Commented out
    df_ml_before_dropna = df_ml.copy() # For debugging
    df_ml = df_ml.dropna()
    # print(f"    DEBUG add_features: Output df_ml shape: {df_ml.shape}, Rows dropped: {len(df_ml_before_dropna) - len(df_ml)}") # MODIFICATION: Commented out
    # if df_ml.empty: # MODIFICATION: Commented out
        # print(f"    DEBUG add_features: df_ml is EMPTY after dropna. Original index range: {df_ml_before_dropna.index.min()} to {df_ml_before_dropna.index.max() if not df_ml_before_dropna.empty else 'N/A'}")
    return df_ml

# --- Function to get starting balance ---
def get_starting_cash_balance(
    db: Client,
    user_id: uuid.UUID,
    anchor_date: date,
    currency: str
) -> float | None:
    """
    Retrieves the latest closing balance from summary_data on or before the anchor date
    for the specified user and currency.
    """
    print(f"🔍 Fetching starting balance for user {user_id}, currency {currency}, on/before {anchor_date}...")
    
    target_summary_table = ""
    if currency.upper() == "CAD":
        target_summary_table = "Summary_output"
    elif currency.upper() == "USD":
        target_summary_table = "Summary_output(USD)"
    else:
        print(f"⚠️ Invalid currency specified for get_starting_cash_balance: {currency}")
        return None

    try:
        response = (
            db.table(target_summary_table) # Use dynamic table name
            .select('"Closing Balance", "Reporting Date"') # Keep quotes if column names have spaces
            .eq("user_id", str(user_id))
            # No currency filter needed here as table is currency-specific
            .lte('"Reporting Date"', anchor_date.strftime('%Y-%m-%d'))
            .order('"Reporting Date"', desc=True)
            .limit(1)
            .execute()
        )

        if response.data:
            latest_balance = response.data[0].get("Closing Balance")
            latest_date = response.data[0].get("Reporting Date")
            print(f"✅ Found starting balance: {latest_balance} from date {latest_date} for table {target_summary_table}")
            return float(latest_balance) if latest_balance is not None else None
        else:
            print(f"⚠️ No closing balance found in {target_summary_table} for user {user_id} on or before {anchor_date}.")
            return None

    except Exception as e:
        print(f"❌ Error fetching starting balance: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        return None

# --- Core Forecasting Function ---
def forecast_transaction_amounts(
    db: Client,
    user_id: uuid.UUID,
    currency: str, # Need currency to filter transactions
    forecast_start_date: date,
    training_window_days: int = 730,
    forecast_horizon_days: int = 35
) -> pd.DataFrame | None:
    """
    Forecasts daily transaction amounts for a specific user and currency.
    - Fetches transaction data from Supabase.
    - Trains a RandomForest model per transaction code (excluding Treasury).
    - Returns a DataFrame with ['Date', 'Forecasted Amount'].
    """
    print(f"🔄 Starting transaction amount forecast for user {user_id}, currency {currency}, starting {forecast_start_date}...")
    all_forecast_rows = []
    try:
        # 1. Fetch relevant transaction data for the user and currency
        print("  Fetching transaction data...")
        # Define date range for fetching historical data needed for training
        # forecast_start_date is the first day OF PREDICTION (e.g., anchor_date + 1 day)
        history_start_date = forecast_start_date - timedelta(days=training_window_days + forecast_horizon_days + 60) # Fetch extra buffer
        print(f"  DEBUG QUERY DATES: history_start_date for query: {history_start_date.strftime('%Y-%m-%d')}, forecast_start_date for query (upper bound, exclusive): {forecast_start_date.strftime('%Y-%m-%d')}")

        response = (
            db.table("Output") # Changed from transaction_data to Output
            .select('"Reporting Date", "Eff Date", "Transaction Amount", "Transaction Type", "Transaction Code", "transaction_category", "currency"') # Ensure column names match new table
            .eq("user_id", str(user_id))
            .eq("currency", currency.upper()) # Filter by currency
            .gte('"Reporting Date"', history_start_date.strftime('%Y-%m-%d'))
            .lt('"Reporting Date"', forecast_start_date.strftime('%Y-%m-%d'))
            .order('"Reporting Date"', desc=False)
            .limit(100000)  # Ensuring correct indentation for the limit
            .execute()
        )

        if not response.data:
            print(f"⚠️ No transaction data found from Supabase query for user {user_id}, currency {currency} between {history_start_date.strftime('%Y-%m-%d')} and {forecast_start_date.strftime('%Y-%m-%d')}.")
            return None

        df_transactions = pd.DataFrame(response.data)
        print(f"  Fetched {len(df_transactions)} transaction records from Supabase.")
        if not df_transactions.empty:
            df_transactions['Reporting Date'] = pd.to_datetime(df_transactions['Reporting Date'])
            print(f"  DEBUG DF_TRANSACTIONS from Supabase: Min Reporting Date: {df_transactions['Reporting Date'].min().strftime('%Y-%m-%d') if not df_transactions.empty else 'N/A'}, Max Reporting Date: {df_transactions['Reporting Date'].max().strftime('%Y-%m-%d') if not df_transactions.empty else 'N/A'}")
            print(f"  DEBUG DF_TRANSACTIONS from Supabase: Sample data (head 1):")
            print(df_transactions.head(1).to_string())
            print(f"  DEBUG DF_TRANSACTIONS from Supabase: Sample data (tail 1):")
            print(df_transactions.tail(1).to_string())
        else:
            print(f"  DEBUG DF_TRANSACTIONS from Supabase: DataFrame is empty after fetching.")


        # MODIFICATION: Removed some less critical logs from here to reduce verbosity
        # print(f"    DEBUG DF_TRANSACTIONS: Shape BEFORE custom drop_duplicates: {df_transactions.shape}") 
        # Define columns that constitute a unique transaction based on typical financial transaction data
        # These should match the main data columns used for grouping and feature engineering, excluding IDs or timestamps of upload.
        # unique_defining_cols = ['Reporting Date', 'Eff Date', 'Transaction Amount', 'Transaction Type', 'Transaction Code', 'transaction_category']
        # Ensure all specified columns actually exist in df_transactions to avoid KeyError
        # Based on the select query, these columns are: "Reporting Date", "Transaction Amount", "Transaction Code", "transaction_category", "Eff Date", "Transaction Type"
        
        # df_transactions = df_transactions.drop_duplicates(subset=unique_defining_cols, keep='first') # <-- MODIFICATION: Commented out to align with Notebook potentially not doing this exact drop
        
        # print(f"    DEBUG DF_TRANSACTIONS: Shape after (potentially skipped) drop_duplicates: {df_transactions.shape}")


        # DEBUG: Check for duplicates and specific transaction code before any processing
        # print(f"    DEBUG DF_TRANSACTIONS: Shape after drop_duplicates: {df_transactions.drop_duplicates().shape}") # This line might be too resource intensive if df_transactions is large

        # Convert data types
        df_transactions['Reporting Date'] = pd.to_datetime(df_transactions['Reporting Date'])
        df_transactions['Transaction Amount'] = pd.to_numeric(df_transactions['Transaction Amount'], errors='coerce')
        df_transactions = df_transactions.dropna(subset=['Reporting Date', 'Transaction Amount', 'Transaction Code'])

        # Filter out Treasury transactions BEFORE grouping
        print(f"  DEBUG: Shape of df_transactions before Treasury filter: {df_transactions.shape}")
        df_transactions = df_transactions[df_transactions["transaction_category"] != "Treasury"].copy()
        print(f"  DEBUG: Shape of df_transactions after Treasury filter: {df_transactions.shape}")

        if df_transactions.empty:
             print("⚠️ No non-Treasury transaction data found for forecasting.")
             return None

        # DEBUG: Log stats after processing
        if not df_transactions.empty:
            print(f"  DEBUG: Transaction Amount stats AFTER processing (before daily_by_code): {df_transactions['Transaction Amount'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])}")
            print(f"  DEBUG: Sample Transaction Amounts AFTER processing: {df_transactions['Transaction Amount'].head().tolist()}")
        else:
            print("  DEBUG: df_transactions is EMPTY AFTER processing (before daily_by_code).")

        # 2. Aggregate daily amounts per transaction code
        daily_by_code = df_transactions.groupby(['Reporting Date', 'Transaction Code'])['Transaction Amount'].sum().unstack(fill_value=0)
        daily_by_code.index = pd.to_datetime(daily_by_code.index)

        if daily_by_code.empty:
             print("⚠️ Aggregated daily data (daily_by_code) is empty.")
             return None
        
        print(f"  DEBUG: daily_by_code shape: {daily_by_code.shape}, Codes found: {daily_by_code.columns.tolist()}")
        if not daily_by_code.empty:
            print(f"  DEBUG: daily_by_code index range: {daily_by_code.index.min()} to {daily_by_code.index.max()}")

        # 3. Forecast per code
        print(f"  Forecasting for {len(daily_by_code.columns)} transaction codes...")

        # --- Define dates for forecasting ---
        # forecast_dates in the notebook started from forecast_start_date + 1 day for prediction
        # but our loop will iterate from forecast_start_date if we want to include it,
        # or adjust the range if the first prediction is for forecast_start_date + 1 day.
        # The notebook's future_dates = pd.date_range(start=forecast_start_date + timedelta(days=1), periods=forecast_horizon_days, freq='D')
        # Let's align with the notebook: predictions start for the day *after* forecast_start_date.
        
        # Correction: Loop should start from forecast_start_date to include it in the prediction range for daily_total_forecast
        prediction_start_actual = forecast_start_date 
        future_dates_for_loop = pd.date_range(
            start=prediction_start_actual, 
            periods=forecast_horizon_days, 
            freq='D'
        )

        for code in daily_by_code.columns:
            try:
                # print(f"  PROCESSING CODE: {code}") # MODIFICATION: Commented out for now, can be re-enabled if needed for specific code
                # if not daily_by_code[code].empty: # MODIFICATION: Commented out
                    # print(f"    DEBUG DAILY_BY_CODE_STATS: Code {code}, stats before asfreq:\n{daily_by_code[code].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])}")
                    # print(f"    DEBUG DAILY_BY_CODE_SAMPLE: Code {code}, sample before asfreq (5): {daily_by_code[code].dropna().head().tolist()}") 
                    # print(f"    DEBUG DAILY_BY_CODE_INDEX: Code {code}, index range before asfreq: {daily_by_code[code].index.min()} to {daily_by_code[code].index.max()}")
                # else: # MODIFICATION: Commented out
                    # print(f"    DEBUG DAILY_BY_CODE_STATS: Code {code}, daily_by_code[code] is EMPTY before asfreq.")

                series = daily_by_code[code].asfreq('D', fill_value=0)
                # print(f"    DEBUG SERIES: Code {code}, shape after asfreq('D'): {series.shape}, index range: {series.index.min()} to {series.index.max() if not series.empty else 'N/A'}") # MODIFICATION: Commented out
                
                df_ml = pd.DataFrame({'Amount': series})
                df_ml = add_features(df_ml.copy()) # Added .copy() to avoid SettingWithCopyWarning
                df_ml = df_ml.dropna() # Drop rows with NaN from lags/rolling features

                # Define training window - ALIGNED WITH NOTEBOOK LOGIC
                # forecast_start_date is the first day OF PREDICTION (e.g., anchor_date + 1 day from notebook's perspective)
                # So, the notebook's anchor_date is forecast_start_date - 1 day
                notebook_anchor_date = forecast_start_date - timedelta(days=1)

                # Notebook trains on data: [anchor_date - TWD, anchor_date - 1 day]
                # which is [notebook_anchor_date - TWD, notebook_anchor_date - 1 day]
                # or in Python terms: data < notebook_anchor_date
                train_cutoff_date = notebook_anchor_date - timedelta(days=training_window_days)
                
                # Convert python dates to pandas Timestamps for comparison with DatetimeIndex
                pd_train_cutoff_date = pd.Timestamp(train_cutoff_date)
                pd_notebook_anchor_date = pd.Timestamp(notebook_anchor_date)

                # Filter for training data: index >= cutoff and index < notebook_anchor_date
                train = df_ml[(df_ml.index >= pd_train_cutoff_date) & (df_ml.index < pd_notebook_anchor_date)]

                # Log training data details
                if not train.empty:
                    print(f"    DEBUG TRAIN_DF: Code {code}, shape after applying window: {train.shape}, index range: {train.index.min() if not train.empty else 'N/A'} to {train.index.max() if not train.empty else 'N/A'}")
                    if not train['Amount'].empty:
                        print(f"    DEBUG Y_TRAIN_STATS: Code {code}, y_train stats:\n{train['Amount'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])}")
                        print(f"    DEBUG Y_TRAIN_SAMPLE: Code {code}, y_train sample (5): {train['Amount'].head().tolist()}")
                    else:
                        print(f"    DEBUG TRAIN_DATA: Code {code}, y_train is EMPTY.")

                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(train.drop(columns='Amount'), train['Amount'])

                # Forecast loop
                last_known_df = df_ml.copy() # Use the full df_ml (with history and NaNs from features) for creating future features.
                
                prediction_dates = pd.date_range(start=forecast_start_date, periods=forecast_horizon_days, freq='D')
                # The prediction_dates should start from forecast_start_date (which is anchor_date + 1 day), this is correct.

                for current_pred_date in prediction_dates:
                    # Prepare the feature row for prediction
                    temp_row_for_features = pd.DataFrame(index=[current_pred_date])
                    temp_row_for_features['Amount'] = 0 # Placeholder, real value doesn't matter for feature calculation here

                    # Combine with last_known_df to correctly calculate features (esp. lag1) for date_to_predict
                    # then take only the row for the date_to_predict. add_features handles dropna
                    # Ensure last_known_df index is datetime
                    if not isinstance(last_known_df.index, pd.DatetimeIndex):
                        last_known_df.index = pd.to_datetime(last_known_df.index)

                    combined_for_single_pred_features = pd.concat([last_known_df, temp_row_for_features])
                    
                    # We need to be careful here: add_features might drop the row if lag1 is NaN
                    # For the very first prediction, lag1 will be based on the last actual data point.
                    # For subsequent predictions, lag1 will be based on the previous day's prediction.
                    
                    features_for_this_day_df = add_features(combined_for_single_pred_features.assign(Amount=combined_for_single_pred_features['Amount'].fillna(0))).loc[[current_pred_date]]
                    
                    if features_for_this_day_df.empty:
                        print(f"    DEBUG: Code {code} - Skipping date {current_pred_date.strftime('%Y-%m-%d')} due to empty features after add_features.")
                        # Decide how to handle this: predict 0, or skip and potentially break cumsum?
                        # For now, let's predict 0 if features are missing, to maintain continuity for cumsum.
                        y_pred_single = 0.0 
                    else:
                        X_pred_single = features_for_this_day_df.drop(columns='Amount', errors='ignore')
                        print(f"      DEBUG PREDICT: Code {code}, Date {current_pred_date.strftime('%Y-%m-%d')}, X_pred_single shape: {X_pred_single.shape}, Columns: {X_pred_single.columns.tolist()}")
                        if X_pred_single.empty and 'Amount' not in features_for_this_day_df.columns : # if only Amount column existed
                             y_pred_single = 0.0 # or handle as error
                        elif X_pred_single.empty and 'Amount' in features_for_this_day_df.columns and len(features_for_this_day_df.columns) ==1 :
                             y_pred_single = 0.0 # only amount column, so predict 0
                        else:
                            y_pred_single = model.predict(X_pred_single)[0]

                    # Ensure forecasted amount is 0 for weekends
                    # original_pred_for_debug = y_pred_single # Store before weekend modification for logging
                    # is_weekend_day_debug = False
                    # if current_pred_date.dayofweek >= 5: # 5 for Saturday, 6 for Sunday
                    #     y_pred_single = 0.0
                    #     is_weekend_day_debug = True
                    
                    # if is_weekend_day_debug: # Log specifically for weekends
                    #     print(f"    DEBUG WEEKEND: Code {code}, Date {current_pred_date.strftime('%Y-%m-%d')}, Orig Pred: {original_pred_for_debug:.2f}, Final Pred: {y_pred_single:.2f}")

                    all_forecast_rows.append({
                        'Date': current_pred_date,
                        'Forecasted Amount': y_pred_single,
                        'Transaction Code': code # Keep track of code for potential debugging or more granular output
                    })

                    # Update last_known_df with the new prediction to be used for the next day's lag1
                    # Need to create a proper DataFrame row with all features for the prediction.
                    # The easiest way is to append the prediction and then re-run add_features on the relevant part.
                    # However, for efficiency and directness as in the notebook:
                    new_predicted_row = pd.DataFrame({'Amount': [y_pred_single]}, index=[current_pred_date])
                    
                    # Add features to this new row based on its own values and the context of last_known_df (for lags)
                    # This is tricky. The notebook's `row = add_features(pd.concat([last_known, row])).iloc[-1:]` was key.
                    # Let's replicate that logic for updating last_known_df more directly.
                    # Construct the row that *would have been* if y_pred_single was its 'Amount'
                    
                    # To update last_known_df correctly for the *next* iteration's lag,
                    # we need to append the raw prediction and then, if we were to re-calculate features,
                    # that 'Amount' would be used.
                    # The notebook did: last_known = pd.concat([last_known, new_row])
                    # where new_row was just {'Amount': [y_pred]}, index=[date]
                    # This implies that add_features IS ROBUST to being called on a df that grows and has its features recomputed.

                    # For simplicity and to match notebook's update of `last_known`:
                    # Add the predicted amount with its date to last_known_df.
                    # The features for this new row aren't immediately needed for `last_known_df`'s *own* consistency,
                    # but the 'Amount' at this date is crucial for the *next* day's lag1 calculation.
                    
                    # If 'Amount' column already has features computed, we need to handle this carefully.
                    # The notebook's last_known seems to be a DataFrame that already has features.
                    # When it does `pd.concat([last_known, new_row_with_only_amount])`, then `add_features`
                    # recalculates features for the whole thing.

                    # Let's ensure last_known_df is just 'Amount' and features are added on the fly for prediction
                    # This means df_ml_original should be the source for historical features.
                    # And last_known_for_lags should be a simple series/df of 'Amount'
                    
                    # Re-thinking the loop structure to be closer to notebook:
                    # last_known_df should accumulate the predictions and their *original* features.
                    # The notebook did:
                    #  last_known = df_ml.copy() # df_ml has features
                    #  for date in future_dates:
                    #    row = pd.DataFrame(index=[date])
                    #    row['Amount'] = 0
                    #    row_with_features = add_features(pd.concat([last_known, row])).iloc[-1:] # Calculate features for THIS day
                    #    X_pred = row_with_features.drop(columns='Amount')
                    #    y_pred = model.predict(X_pred)[0]
                    #    ... append y_pred ...
                    #    new_row_with_prediction = pd.DataFrame({'Amount': [y_pred]}, index=[date]) # NO other features here
                    #    last_known = pd.concat([last_known, new_row_with_prediction]) # Add raw prediction to last_known
                    # This means `last_known` becomes a mix of historical rows (with full features)
                    # and future predicted rows (initially with only 'Amount', features re-calculated on the fly by add_features).

                    # Let's refine `last_known_df` update:
                    # It should store the predicted 'Amount' and the *features that were used to make that prediction*.
                    # No, the notebook's `last_known` update is simpler: it just appends the new 'Amount'.
                    # The `add_features(pd.concat([last_known, row]))` part re-computes features on this growing dataframe.
                    
                    # Sticking to the provided web app structure for X_pred_full and y_pred (predict all at once)
                    # The weekend check is already in place for that.
                    # The main issue was likely the iterative nature of prediction for lags not being present.
                    # The current web app code *already* tries to build features for all future dates at once.
                    # Let's ensure the lag features are correctly propagated if we stick to non-iterative.
                    # The problem with non-iterative: lag1 for day T+2 uses actual from T, not prediction from T+1.

                    # Reverting to ITERATIVE to match notebook for lag propagation:
                    # (The code above this comment block was already iterative in prediction but not in last_known update for next lag)
                    # The block below re-implements the iterative prediction and `last_known` update for lags.
                    
                    # This was the original structure I was aiming for in the edit_file call.
                    # The code block above this was an attempt to fix the apply_model's misplacement.
                    # Let's assume the prior section (predicting y_pred_single) is correct and focus on updating last_known_df

                    # To update last_known_df for the *next* iteration's lag calculation:
                    # We append the row used for prediction, but with the 'Amount' updated to y_pred_single.
                    # features_for_this_day_df already has the correct features (dayofweek, month, etc.)
                    # and its 'lag1' was based on the previous state of last_known_df.
                    
                    update_row = features_for_this_day_df.copy()
                    update_row['Amount'] = y_pred_single
                    
                    # Ensure last_known_df's index is DatetimeIndex before concatenating
                    if not isinstance(last_known_df.index, pd.DatetimeIndex):
                        last_known_df.index = pd.to_datetime(last_known_df.index)
                    if not isinstance(update_row.index, pd.DatetimeIndex):
                        update_row.index = pd.to_datetime(update_row.index)

                    # Remove the date from last_known_df if it exists (e.g. if it was a placeholder from df_ml_original)
                    if current_pred_date in last_known_df.index:
                        last_known_df = last_known_df.drop(current_pred_date)
                    
                    last_known_df = pd.concat([last_known_df, update_row])
                    last_known_df = last_known_df.sort_index() # Keep it sorted by date
                    print(f"      DEBUG UPDATE_LAST_KNOWN: Code {code}, Date {current_pred_date.strftime('%Y-%m-%d')}, last_known_df updated. Shape: {last_known_df.shape}, Tail(1):\n{last_known_df.tail(1)}")

                # This was the loop for a single code.
                # `all_forecast_rows` collects predictions from all codes.

            except Exception as e_code:
                print(f"⚠️ Error forecasting for code {code}: {type(e_code).__name__} - {str(e_code)}")
                traceback.print_exc()
                continue

        if not all_forecast_rows:
             print("⚠️ No forecast rows generated for any code.")
             return None

        # 4. Aggregate forecasts per day
        forecast_df = pd.DataFrame(all_forecast_rows)
        # Ensure Date column is present before grouping
        if 'Date' not in forecast_df.columns:
             print("❌ 'Date' column missing in forecast rows before grouping.")
             return None
             
        daily_total_forecast = forecast_df.groupby('Date')['Forecasted Amount'].sum().reset_index()

        print(f"✅ Transaction amount forecast completed. Shape: {daily_total_forecast.shape}")
        return daily_total_forecast

    except Exception as e:
        print(f"❌ Error in forecast_transaction_amounts: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        return None

# --- Function to build cash position ---
def build_cash_position_forecast(
    db: Client, 
    user_id: uuid.UUID,
    currency: str,
    daily_forecast_df: pd.DataFrame,
    starting_cash_balance: float,
    forecast_start_date: date
) -> pd.DataFrame | None:
    """
    Builds the cash position forecast using daily forecasted amounts and starting balance.
    Fetches actual closing balances from summary_data for comparison.
    """
    if daily_forecast_df is None or daily_forecast_df.empty:
        print("❌ Cannot build cash position: daily_forecast_df is empty.")
        return None
    if starting_cash_balance is None:
        print("❌ Cannot build cash position: starting_cash_balance is None.")
        return None

    print("🏗️ Building cash position forecast...")
    try:
        daily_forecast_df['Date'] = pd.to_datetime(daily_forecast_df['Date'])
        daily_forecast_df = daily_forecast_df.sort_values(by='Date').reset_index(drop=True)

        daily_forecast_df['Cumulative Forecast Amount'] = daily_forecast_df['Forecasted Amount'].cumsum()
        daily_forecast_df['Forecasted Cash Balance'] = starting_cash_balance + daily_forecast_df['Cumulative Forecast Amount']

        print(f"  Fetching actual closing balances for comparison (User: {user_id}, Currency: {currency})...")
        
        target_summary_table_actuals = ""
        if currency.upper() == "CAD":
            target_summary_table_actuals = "Summary_output"
        elif currency.upper() == "USD":
            target_summary_table_actuals = "Summary_output(USD)"
        else:
            print(f"⚠️ Invalid currency specified for fetching actuals in build_cash_position_forecast: {currency}")
            # Decide if we proceed with an empty actuals_df or return None/raise error
            actuals_df = pd.DataFrame(columns=['Date', 'Actual Cash Balance'])
            # Ensure 'Date' is datetime if we proceed with empty
            actuals_df['Date'] = pd.to_datetime(actuals_df['Date'])
            # For safety, let's return None if currency is invalid as it affects core logic
            # return None 
            # Or, allow it to proceed, and the merge will just not find actuals.
            # Let's proceed but log the warning heavily.

        actual_balances_response = None
        if target_summary_table_actuals: # Only query if table name is valid
            try:
                # Fetch all relevant actual balances around the forecast period for merging
                # We need balances for dates that are in daily_forecast_df['Date']
                min_forecast_date = daily_forecast_df['Date'].min()
                max_forecast_date = daily_forecast_df['Date'].max()

                actual_balances_response = (
                    db.table(target_summary_table_actuals)
            .select('"Reporting Date", "Closing Balance"')
            .eq("user_id", str(user_id))
                    # No currency filter as table is specific
                    .gte('"Reporting Date"', min_forecast_date.strftime('%Y-%m-%d'))
                    .lte('"Reporting Date"', max_forecast_date.strftime('%Y-%m-%d'))
            .order('"Reporting Date"', desc=False)
            .execute()
        )
            except Exception as e_actuals:
                print(f"❌ Error fetching actual balances from {target_summary_table_actuals}: {e_actuals}")
                # Proceed with empty actuals if DB query fails, or handle as critical error
                actual_balances_response = None # Ensure it's None to fall through to empty df

        if actual_balances_response and actual_balances_response.data:
            actuals_df = pd.DataFrame(actual_balances_response.data)
            actuals_df.rename(columns={"Reporting Date": "Date", "Closing Balance": "Actual Cash Balance"}, inplace=True)
            actuals_df['Date'] = pd.to_datetime(actuals_df['Date'])
            actuals_df['Actual Cash Balance'] = pd.to_numeric(actuals_df['Actual Cash Balance'], errors='coerce')
            print(f"  Fetched {len(actuals_df)} actual balance records from {target_summary_table_actuals}.")
            
            # MODIFICATION: Add .ffill() to Actual Cash Balance like in the notebook
            actuals_df['Actual Cash Balance'] = actuals_df['Actual Cash Balance'].ffill()
            print(f"  Applied .ffill() to 'Actual Cash Balance'. Nulls after ffill: {actuals_df['Actual Cash Balance'].isnull().sum()}")
        else:
            print(f"⚠️ No actual closing balances found for user {user_id}, currency {currency} in {target_summary_table_actuals} for the forecast period, or error fetching.")
            actuals_df = pd.DataFrame(columns=['Date', 'Actual Cash Balance'])
            # Ensure 'Date' is datetime if we proceed with empty
            actuals_df['Date'] = pd.to_datetime(actuals_df['Date'])

        # 3. Merge daily forecast with actual closing balances
        # Ensure daily_forecast_df['Date'] is also datetime before merging
        daily_forecast_df['Date'] = pd.to_datetime(daily_forecast_df['Date'])
        merged_df = pd.merge(daily_forecast_df, actuals_df, on="Date", how="left")

        # Use standard pandas column names and rename only before insertion if needed
        # Ensure 'Date' column exists in merged_df (it should from daily_forecast_df)
        if 'Date' not in merged_df.columns:
            print("❌ Critical error: 'Date' column is missing from merged_df before final selection.")
            return None # Or raise an error

        final_df = merged_df[['Date', 'Forecasted Amount', 'Forecasted Cash Balance', 'Actual Cash Balance']].copy()
        print(f"✅ Cash position forecast built. Shape: {final_df.shape}")
        return final_df

    except Exception as e:
        print(f"❌ Error in build_cash_position_forecast: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        return None

# --- Main Orchestration Function ---
async def generate_forecast_for_user(
    db: Client,
    user_id: uuid.UUID,
    currency: str,
    forecast_anchor_date: Optional[date] = None,
    training_window_days: int = 730,
    forecast_horizon_days: int = 35
) -> Optional[pd.DataFrame]:
    """
    Generates a full cash forecast (transaction amounts and cash position) for a user.
    Returns a DataFrame ready for insertion into 'full_forecast_output' table,
    or None if a critical error occurs.
    """
    print(f"🚀 Starting full forecast generation for user: {user_id}, currency: {currency}, anchor: {forecast_anchor_date}")

    # 1. Determine the actual start date for the forecast (day after anchor or day after today)
    if forecast_anchor_date:
        actual_forecast_start_date = forecast_anchor_date + timedelta(days=1)
    else:
        actual_forecast_start_date = date.today() + timedelta(days=1)
    print(f"  Forecast will start from: {actual_forecast_start_date}")

    # 2. Get Starting Cash Balance as of the forecast_anchor_date (or today if anchor is None)
    balance_anchor_date = forecast_anchor_date if forecast_anchor_date else date.today()
    starting_balance = get_starting_cash_balance(db, user_id, balance_anchor_date, currency)
    
    if starting_balance is None:
        print(f"❌ Critical: Could not retrieve starting cash balance for {currency} as of {balance_anchor_date}. Aborting forecast.")
        return None
    print(f"  Using starting cash balance: {starting_balance} for {currency} as of {balance_anchor_date}")

    # 3. Forecast Daily Transaction Amounts
    daily_transaction_forecast_df = forecast_transaction_amounts(
            db=db,
            user_id=user_id,
            currency=currency,
        forecast_start_date=actual_forecast_start_date, # This is the first day of *prediction*
            training_window_days=training_window_days,
            forecast_horizon_days=forecast_horizon_days
        )
    if daily_transaction_forecast_df is None or daily_transaction_forecast_df.empty:
        print(f"❌ Critical: Failed to generate daily transaction forecasts for {currency}. Aborting forecast.")
        return None
    print(f"  Generated daily transaction forecast. Shape: {daily_transaction_forecast_df.shape}")

    # 4. Build Cash Position Forecast
    cash_position_df = build_cash_position_forecast(
            db=db,
            user_id=user_id,
        currency=currency, # Pass currency for fetching actuals for comparison
        daily_forecast_df=daily_transaction_forecast_df, # This contains 'Date' and 'Forecasted Amount'
            starting_cash_balance=starting_balance,
        forecast_start_date=actual_forecast_start_date # Used for fetching comparable actuals
    )
    if cash_position_df is None or cash_position_df.empty:
        print(f"❌ Critical: Failed to build cash position forecast for {currency}. Aborting forecast.")
        return None
    print(f"  Built cash position forecast. Shape: {cash_position_df.shape}")

    # 5. Prepare the final DataFrame for 'full_forecast_output' table
    # Expected columns: Date, Forecasted Amount, Forecasted Cash Balance, Actual Cash Balance
    # cash_position_df should already contain: Date, Forecasted Amount, Forecasted Cash Balance, Actual Cash Balance
    
    final_forecast_df = cash_position_df.copy()
    
    # Ensure all required columns are present and correctly named as per 'full_forecast_output'
    # Rename columns if necessary, though build_cash_position_forecast should already align them.
    # For example, if build_cash_position_forecast produced 'Forecasted_Cash_Balance', rename it here.
    # Assuming build_cash_position_forecast output is already: 
    # ['Date', 'Forecasted Amount', 'Forecasted Cash Balance', 'Actual Cash Balance']

    # Add user_id (will be done in the router before insertion, but good to note)
    # final_forecast_df['user_id'] = str(user_id) 
        
    # Convert Date to YYYY-MM-DD string if it's not already, for DB insertion consistency
    # (though Supabase client might handle datetime objects correctly)
    if 'Date' in final_forecast_df.columns and pd.api.types.is_datetime64_any_dtype(final_forecast_df['Date']):
        final_forecast_df['Date'] = final_forecast_df['Date'].dt.strftime('%Y-%m-%d')

    # Ensure numeric columns are rounded (e.g., to 2 decimal places)
    numeric_cols = ['Forecasted Amount', 'Forecasted Cash Balance', 'Actual Cash Balance']
    for col in numeric_cols:
        if col in final_forecast_df.columns:
            final_forecast_df[col] = pd.to_numeric(final_forecast_df[col], errors='coerce').round(2)
            # Handle potential NaNs introduced by to_numeric if needed (e.g., fill with None for DB)
            # final_forecast_df[col] = final_forecast_df[col].astype(object).where(pd.notnull(final_forecast_df[col]), None)

    # Select only the columns for the final table in the correct order if desired
    output_columns = ['Date', 'Forecasted Amount', 'Forecasted Cash Balance', 'Actual Cash Balance']
    # Filter out any columns that might not exist if some steps failed to produce them fully
    existing_output_columns = [col for col in output_columns if col in final_forecast_df.columns]
    final_forecast_df = final_forecast_df[existing_output_columns]

    print(f"✅ Final forecast DataFrame for user {user_id}, currency {currency} generated. Shape: {final_forecast_df.shape}. Columns: {final_forecast_df.columns.tolist()}")
    
    # Add the currency column to the DataFrame
    final_forecast_df['currency'] = currency.upper() # Or just currency if lowercase is preferred and consistent
    print(f"  Added 'currency' column with value: {currency.upper()}. Shape after adding currency: {final_forecast_df.shape}")

    # Ensure specific column names as expected by the database/Pydantic model
    # Rename columns if necessary, though build_cash_position_forecast should already align them.
    # For example, if build_cash_position_forecast produced 'Forecasted_Cash_Balance', rename it here.
    # Assuming build_cash_position_forecast output is already: 
    # ['Date', 'Forecasted Amount', 'Forecasted Cash Balance', 'Actual Cash Balance']

    # Add user_id (will be done in the router before insertion, but good to note)
    # final_forecast_df['user_id'] = str(user_id) 
    
    # Convert Date to YYYY-MM-DD string if it's not already, for DB insertion consistency
    # (though Supabase client might handle datetime objects correctly)
    if 'Date' in final_forecast_df.columns and pd.api.types.is_datetime64_any_dtype(final_forecast_df['Date']):
        final_forecast_df['Date'] = final_forecast_df['Date'].dt.strftime('%Y-%m-%d')

    # Ensure numeric columns are rounded (e.g., to 2 decimal places)
    numeric_cols = ['Forecasted Amount', 'Forecasted Cash Balance', 'Actual Cash Balance']
    for col in numeric_cols:
        if col in final_forecast_df.columns:
            final_forecast_df[col] = pd.to_numeric(final_forecast_df[col], errors='coerce').round(2)
            # Handle potential NaNs introduced by to_numeric if needed (e.g., fill with None for DB)
            # final_forecast_df[col] = final_forecast_df[col].astype(object).where(pd.notnull(final_forecast_df[col]), None)

    # Select only the columns for the final table in the correct order if desired
    output_columns = ['Date', 'Forecasted Amount', 'Forecasted Cash Balance', 'Actual Cash Balance']
    # Filter out any columns that might not exist if some steps failed to produce them fully
    existing_output_columns = [col for col in output_columns if col in final_forecast_df.columns]
    final_forecast_df = final_forecast_df[existing_output_columns]

    print(f"✅ Forecast generation successful for {currency}. Final DataFrame shape: {final_forecast_df.shape}. Columns: {final_forecast_df.columns.tolist()}")
    
    # The actual DB insertion will be handled by the calling API endpoint.
    # The old DB interaction for 'forecast_results' table (delete and insert) is removed from here.
    
    return final_forecast_df

# Placeholder for the FastAPI endpoint that will call this. 