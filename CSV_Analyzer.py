# Standard library module
import csv
import logging
import re
import os

# Third party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import re
import seaborn as sns
import streamlit as st

# Initialize session state for showing graph options
if "show_graph_options" not in st.session_state:
    st.session_state.show_graph_options = False

# Set up the general Streamlit UI
st.title("CSV Analyzer")
st.header("Analyze your CSV files with fun!")

# A. Upload the CSV file
uploaded_file = st.file_uploader("Choose a CSV File")

# Initialize 'df', so that 'df' is globally available 
df = None

# Initialize a flag to show warning or info or error messages when the user presses the "Show Graphical Representation" button
show_messages_graph = False

# Clean the timedelta column for better results


def clean_timedelta(timedelta_string):
    """Cleans a timedelta string."""
    # Checks if the "timedelta_string" is a missing value
    if pd.isna(timedelta_string):
        return pd.NaT

    # Convert the string into lower case and remove any leading or trailing whitespace
    cleaned_timedelta_string = str(timedelta_string).lower().strip()
    # Remove extra spaces
    cleaned_timedelta_string = " ".join(cleaned_timedelta_string.split())
    
    # Extracting 'days'
    # "\d+": Matches one or more digits
    # The parenthesis () around the "\d+" creates a capturing group. The part of the string that matches this group can be extracted later.
    # "\s*": Matches zero or more whitespace charecters
    # "days?": Matches 'day' or 'days' (the '?' makes the 's' optional)
    day_match = re.search(r'(\d+)\s*d(?:ays?)?\s*', cleaned_timedelta_string)
    # If a match is found it extracts the captured group using "day_match.group(1)"; if no match is found, "day" is set to 0
    day = int(day_match.group(1)) if day_match else 0
    # "re.sub(pattern, replacement, string)" - finds all occurences of a pattern in a string and replaces them with a specified replacement string
    cleaned_timedelta_string = re.sub(r'(\d+)\s*d(?:ays?)?\s*', '', cleaned_timedelta_string).strip()
                                      
    # Extracting 'hours', 'mins', and 'secs'                                  
    # Understanding "(?:hours?|:)?":
    #   "(?:....)" - this is a non-capturing group; used to gruop parts of the regex without creating a separate capturing group
    #   "hours?": Matches 'hour' or 'hours'
    #   "|": This is the OR operator
    #   ":": Matches a literal colon
    #   "?" at the end makes the entire non-capturing group optional. So, it can match 'hour', 'hours', ':', or nothing
    # The "*" in "\d*" means that it can even match an empty string if minutes are not explicitly present
    # The "(?:mins?|:)?" similar to the hours part, matches 'min', 'mins', ':', or nothing
    # "(\d*\.?d*)" - Matches and captures the seconds part
    #   Where "\.?" captures a literal decimal point
    time_parts = re.findall(r'(\d+)?\s*(?:hours?|hrs?|h|:)?\s*(\d*)?\s*(?:mins?|minutes?|m|:)?\s*(\d*\.?d*)?\s*(?:secs?|seconds?|s)?', cleaned_timedelta_string)
    
    # Extract hours, minutes, and seconds from the matched time parts
    # If "re.findall" finds any matches, "time_parts[0][0]" takes the first match, then:
    # Extracts the first captured group (i.e. hours), converts it to an integer, and assigns it to a variable "hours". Otherwise "hours" is set to 0
    hours = int(time_parts[0][0]) if time_parts and time_parts[0][0] else 0
    # Similarly it captures and extracts the second captured group (i.e. minutes), converts it to an integer, and assign it to a variable
    minutes = int(time_parts[0][1]) if time_parts and time_parts[0][1] else 0
    # It does the same for the third captured group (seconds)
    seconds = float(time_parts[0][2]) if time_parts and time_parts[0][2] else 0

    # Handling single numbers (Assuming that single numbers represent seconds)
    if not time_parts and re.match(r'^\d+\.?\d*$', cleaned_timedelta_string):
        seconds = float(cleaned_timedelta_string)
                                      
    # Standardize the time format
    time_string = f"{hours:02}:{minutes:02}:{seconds:06.3f}"
    if day > 0:
        standardize_string = f"{day} days {time_string}"
    else:
        standardize_string = time_string
    
    return standardize_string


def convert_to_timedelta(df, col):
    """Converts a column to timedelta and specifies the error."""
    problematic_values = []

    for index, value in df[col].items():
        try:
            cleaned_value = clean_timedelta(value)
            if pd.notna(cleaned_value):
                pd.to_timedelta(cleaned_value)
        except ValueError:
            problematic_values.append(value)

    if problematic_values:
        st.warning(f"The following values in '{col}' could not be converted: {problematic_value}")

    df[col] = df[col].apply(lambda x: pd.to_timedelta(clean_timedelta(x), errors='coerce'))
    return df


# A1. Specify the column types
# Let the user specify the column type
        

def column_specifier():
    """Let the user specify the column type."""
    global df
    
    # Create a copy of the original dataframe
    modified_df = df.copy()
    
    all_columns = df.columns.tolist()

    # Let the user choose the categorical columns
    categorical_column = st.multiselect("Select categorical columns:", all_columns)
    # Let the user choose the numerical columns
    remaining_columns = [col for col in all_columns if col not in categorical_column]
    numerical_column = st.multiselect("Select numerical columns:", remaining_columns)
    # Let the user choose the date/time columns
    remaining_columns_after_numerical = [col for col in remaining_columns if col not in numerical_column]
    datetime_column = st.multiselect("Select date/time columns:", remaining_columns_after_numerical)
    # Let the user choose the boolean columns
    remaining_columns_after_datetime = [col for col in remaining_columns_after_numerical if col not in datetime_column]
    boolean_column = st.multiselect("Select boolean columns:", remaining_columns_after_datetime)
    # Let the user choose the timedelta columns
    remaining_columns_after_boolean = [col for col in remaining_columns_after_datetime if col not in boolean_column]
    timedelta_column = st.multiselect("Select timedelta columns:", remaining_columns_after_boolean)

    # Data Validation - Check for conflicting column type
    all_selections = categorical_column + numerical_column + datetime_column + boolean_column + timedelta_column
    if len(all_selections) != len(set(all_selections)):
        st.error("Error: It seems like you have selected the same column for multiple column types! Please correct this.")
        return # Exit from the function
            
    # Display the selected column types
    st.subheader("Selected Column Types:")
    st.write("Categorical:", categorical_column)
    st.write("Numerical:", numerical_column)
    st.write("Datetime:", datetime_column)
    st.write("Boolean:", boolean_column)
    st.write("Timedelta:", timedelta_column)

    # Apply what the user specified
    # Handling the categorical columns
    for col in categorical_column:
        if col in modified_df.columns:
            modified_df[col] = modified_df[col].astype("category")
                
    # Handling the numerical columns
    for col in numerical_column:
        if col in modified_df.columns:
            try:
                # Remove any non-numeric charecter except for the decimal point
                modified_df[col] = modified_df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
                modified_df[col] = pd.to_numeric(modified_df[col], errors="raise")
            except ValueError as ve:
                st.warning(f"Could not convert column '{col}' to numerical type! It remains as is.\nError: {ve}")
                
    # Handling the date/time columns
    for col in datetime_column:
        if col in modified_df.columns:
            modified_df[col] = pd.to_datetime(modified_df[col], errors="coerce")

    # Handling the boolean columns
    for col in boolean_column:
        if col in modified_df.columns:
            modified_df[col] = (modified_df[col]
                                .astype("str")
                                .str.lower()
                                .str.replace("yes", "true")
                                .str.replace("no", "false")
                                .str.replace("1", "true")
                                .str.replace("0", "false")
                            )
            modified_df[col] = modified_df[col].astype("bool")

    # Handling the timedelta columns
    for col in timedelta_column:
        if col in modified_df.columns:
            modified_df = convert_to_timedelta(modified_df, col)

    # Display the dataframe with user-specified column types
    st.subheader("Dataframe with User-specified Column Types:")
    st.dataframe(modified_df.dtypes.to_frame(name="Data Type"))

    # Update the original 'df'
    df = modified_df.copy()


# Standardize the timedelta units


def standardize_timedelta(df, col, target_unit="seconds"):
    """
    Convert a timedelta column in a dataframe to a specified unit.

    Note: Conversions to "months" and "years" are approximates.
    """
    # Check if the column exists.
    if col not in df.columns:
        st.error(f"Timedelta column '{col}' not found in the dataframe.")
        return None

    # Check the column type
    if not pd.api.types.is_timedelta64_dtype(df[col]):
        st.error(f"Column '{col}' is not a timedelta column.")
        return None

    # Make a dictionary for the units
    conversion_units = {
        "seconds": pd.Timedelta(seconds=1),
        "minutes": pd.Timedelta(minutes=1),
        "hours": pd.Timedelta(hours=1),
        "days": pd.Timedelta(days=1),
        "months": pd.Timedelta(days=30.4375), # Calculating the best average for the "month": (31 * 7 + 30 * 4 + 28 + 29) / 12 = 30.4375
        "years": pd.Timedelta(days=365.25), # Calculating the best average for the "year": ((365 * 3) + (366 * 1)) / 4 = 365.25
    }

    # Check for any unsupported units
    if target_unit not in conversion_units:
        st.error(
            f"Unsupported '{target_unit}'."
            "Please choose from 'seconds', 'minutes', 'hours', 'days', 'months', or 'years'"
        )
        return None

    # Convert the units
    return df[col] / conversion_units[target_unit]


# D. Message handling for graphs


def display_message(df):
    """Show warning, info or error messages (resulting from data validation) only when the "Show Graphical Representation" button is pressed."""
    # I. Category - Data Validation - Check for high cardinality (high cardinality can make some plots less effective)
    for col in df.select_dtypes(include="category").columns:
        if df[col].nunique() > 50: # Adjust the threshold as needed
            st.warning(
                f"Column '{col}' has high cardinality ({df[col].nunique()} unique values). This may impact the effectiveness of certain plots."
                f"\n\nPlease consider summarizing or filtering the data before plotting a graph for a better visualization."
            )

    # II. Numerical - Data Validation - Check for non-numeric values after the conversion (Gives user a chance to clean the data before plotting a graph)
    numerical_col = df.select_dtypes (include=np.number).columns.tolist()
    # Exclude the timedelta column
    col_to_check = [col for col in numerical_col if col not in df.select_dtypes(include="timedelta64[ns]").columns]
    for col in col_to_check:
        non_numeric_values = df[col][pd.to_numeric(df[col], errors="coerce").isna()]
        if not non_numeric_values.empty:
            st.warning(
                f"Non-numeric values found in '{col}': {non_numeric_values.head(10).to_list()} ...."
                f"\n\nPlease consider cleaning or checking the data before plotting a graph."
            )

    # III. DateTime - Data Validation - Check for the NaT values (NaT values can make the graph meaningless)
    for col in df.select_dtypes(include="datetime64").columns:
        num_nat_values = df[col].isna().sum()
        if num_nat_values > 0:
            st.warning(
                f"Warning: Column '{col}' has {num_nat_values} values that could not be converted to datetime."
                f"\n\nPlease consider cleaning or checking the data before plotting a graph."
            )

    # IV. Boolean - Data Validation - Check for the NaN values (NaN values can make the graph meaningless)
    bool_col = df.select_dtypes(include="bool").columns
    for col in bool_col:
        num_nan_values = df[col].isna().sum()
        if num_nan_values > 0:
            st.warning(
                f"Warning: Column '{col}' has {num_nan_values} values that could not be converted to boolean."
                f"\n\nPlease consider handling them for an accurate graph."
            )
        # IV.A. Data Validation - User guide for cleaning inconsistent boolean representation
        # Check for inconsistent Boolean values
        inconsistent_boolean_values = set(df[col].unique()) - {True, False, 1, 0}
        if inconsistent_boolean_values:
            st.info(
                f"Unique values in boolean column '{col}': {unique_values}"
                f"\n\nFor consistent boolean converson, please consider standardizing values to 'True' or 'False' / '1' or '0' before selecting the boolean type."
                f"\n\nFor example: df['{col}'] = df['{col}'].astype(str).str.lower().str.replace('yes', 'true').str.replace('no', 'false')'"
            )

    # V. Timedelta - Data Validation - Check for mixed timedelta units (mixed timedelta units can cause misinterpretation and errors in calculations and comparisoins)
    for col in df.select_dtypes(include="timedelta64[ns]").columns:
        unit_types = set()
        for value in df[col].dropna(): # Iterate over the NaN values
            if isinstance(value, pd.Timedelta):
                unit_types.add(value.components._fields) # Access the timedelta components
        if len(unit_types) > 1:
            st.warning(
                f"Warning: Column '{col}' contains mixed timedelta units, which might lead to misinterpretations!"
                f"\n\nPlease consider cleaning or checking the data before plotting a graph."
            )
        # V.A. Data Validation - Check for the NaT values (NaT values can make the graphs meaningless)
        num_nat_values = df[col].isna().sum()
        if num_nat_values > 0:
            st.warning(f"Warning: Column '{col}' has {num_nat_values} values that could not be converted to timedelta.")


# B. Read the CSV File
try:
    if uploaded_file is not None:
        # Check for the file extension
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            if st.button("Show Preview of the Uploaded File"):
                st.dataframe(df.head()) # Display the first five rows of the CSV file
        else:
            st.error("Please upload a valid CSV file!")
            st.stop()
except Exception as e:
    st.error(
        "Oops! Couldn't read the CSV data."
        f"\n\nDetails: {e}"
    )
    
# A2. Call the "column_specifier()" function
if df is not None:
    column_specifier()

# C. Find any missing values
if df is not None:
    missing_value = df.isnull().sum()
    missing_value_df = pd.DataFrame({"Column": missing_value.index, "Missing Count": missing_value.values})
    st.subheader("Number of Missing Values in the Dataframe:")
    st.dataframe(missing_value_df)
    # C1. Display the options for handling 'missing_value' only if missing values are detected
    if missing_value.sum() > 0:
        st.subheader("Handle the Missing Values")
        
        # Handle the missing values through different techniques
        # Drop NA
        drop_na_options = st.expander("Drop the Missing Values")
        if drop_na_options.checkbox("Drop Rows with Any Missing Values"):
            df.dropna(axis=0, how="any", inplace=True)
            st.write("Rows with the misssing values deleted.")
        if drop_na_options.checkbox("Drop Columns with Any Missing Values"):
            df.dropna(axis=1, how="any", inplace=True)
            st.write("Columns with the misssing values deleted.")
        # Fill NA (General)
        fill_na_options = st.expander("Fill the Missing Values")
        general_fill_value = fill_na_options.text_input("Fill the missing values with a value:", "")
        if general_fill_value:
            # When we use "df.fillna('any_general_fill_value', inplace=True)", pandas tries to fill the missing values with the "any_general_fill_value"
            # But if "any_general_fill_value" is not in the categorical column's defined categories, pandas raises the "....setitem...." TypeError
            # Because it doesn't know how to handle a new, unexpected category.
            for col in df.select_dtypes(include="category").columns:
                if general_fill_value not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([general_fill_value]) # Add a new category
            df.fillna(general_fill_value, inplace=True)
            st.write(f"The missing values are filled with '{general_fill_value}'.")
        # Fill NA (Numerical)
        # 1. Mean
        if fill_na_options.checkbox("Fill the Numerical Missing Values with Mean"):
            numerical_col = df.select_dtypes(include=np.number).columns
            df[numerical_col] = df[numerical_col].fillna(df[numerical_col].mean())
            st.write("The numerical missing values are filled with the mean.")
        # 2. Median
        if fill_na_options.checkbox("Fill the Numerical Missing Values with Median"):
            numerical_col = df.select_dtypes(include=np.number).columns
            df[numerical_col] = df[numerical_col].fillna(df[numerical_col].median())
            st.write("The numerical missing values are filled with the median.")
        # Fill NA (Timedelta)
        # 1. Interpolate
        timedelta_options = st.expander("Fill the 'timedelta' Missing Values")
        if fill_na_options.checkbox("Interpolate the Timedelta Values"):
            timedelta_col = df.select_dtypes(include=pd.Timedelta).columns
            for col in timedelta_col:
                df[col] = df[col].interpolate()
            st.write("The timedelta values are interpolated.")
        # 2. Forward Fill
        if fill_na_options.checkbox("Forward Fill the Missing Values"):
            df.fillna(method="ffill", inplace=True)
            st.write("The missing values are forward filled.")
        # 3. Backward Fill
        if fill_na_options.checkbox("Backward Fill the Missing Values"):
            df.fillna(method="bfill", inplace=True)
            st.write("The missing values are backward filled.")
        # Fill NA (Categorical)
        # 1. Mode
        categorical_fill_options = st.expander("Fill the Categorical Missing Values")
        if categorical_fill_options.checkbox("Fill with the Most Frequent Category"):
            categorical_col = df.select_dtypes(include=["category", "object"]).columns
            for col in categorical_col:
                df[col] = df[col].fillna(df[col].mode()[0])
                # The "[0]" in "mode[0]" is actually referring the "pandas series"'s index, and not the list (series) of mode itself
            st.write("The categorical missing values are filled with the most frequent category.")
        # 2. Fill with the word "Unknown"
        if categorical_fill_options.checkbox("Fill with the Word 'Unknown'"):
            categorical_col = df.select_dtypes(include=["category", "object"]).columns
            for col in categorical_col:
                if pd.api.types.is_categorical_dtype(df[col]): # Check if its a categorical column
                    if "Unknown" not in df[col].cat.categories.to_list(): # Ensure it is a list
                        df[col] = df[col].cat.add_categories(["Unknown"])
                df[col] = df[col].fillna("Unknown")
            st.write("The categorical missing values are filled with the word 'Unknown'.")

        # C2. Display the updated dataframe.
        st.subheader("Dataframe After Handling the Missing Values:")
        st.dataframe(df.head(20))

        # C3. Display the number of missing values in the dataframe after handling the missing values
        if st.button("Missing Value Report"):
            missing_value_ahamv = df.isnull().sum()
            missing_value_df_ahamv = pd.DataFrame({"Column": missing_value_ahamv.index, "Missing Count": missing_value_ahamv.values})
            st.subheader("Summary of Missing Values:")
            st.dataframe(missing_value_df_ahamv)

# Convert the timedelta units
if df is not None:
    timedelta_col = df.select_dtypes(include="timedelta64").columns
    if not timedelta_col.empty:
        st.subheader("Convert the Timedelta Units:")
        col_to_convert = st.selectbox("Select a Timedelta Column to Convert", timedelta_col)
        # "target_unit" - the unit to which the timedelta values should be converted
        target_unit = st.selectbox("Select the Target Unit:", ["seconds", "minutes", "hours", "days", "months", "years"])
        st.info(
            "Target unit: The unit to which the timedelta values should be converted"
            "\n\nPlease note that the conversions to 'months' and 'years' are approximations"
        )

        if st.button("Convert Units"):
            converted_units = standardize_timedelta(df, col_to_convert, target_unit)

            if converted_units is not None:
                df[col_to_convert + f"_in_{target_unit}"] = converted_units
                st.success(
                    f"Column '{col_to_convert}' converted to '{target_unit}'. "
                    f"(Approximate for months or years)"
                    f"\n\nA new column '{col_to_convert}_in_{target_unit}' has been created."
                )

                # Show a preview of the updated dataframe
                st.subheader("Dataframe After Handling the Timedelta Units:")
                st.dataframe(df.head())
                
# D1. Display relevant warnings, errors, or infos to the user before generating the visualization 
if df is not None:
    # Ask if the user wants a graphical representation
    if st.button("Show Graphical Representation"):
        show_messages_graph = True
        st.session_state.show_graph_options = True # Update the session state
    # Display the required messages only if the "Show Graphical Representation" button is pressed and 'df' is valid
    if show_messages_graph and df is not None:
        display_message(df) # Call the "display_message(df)"

# D2. Display the graph
if st.session_state.show_graph_options and df is not None:
    graph_type = st.selectbox("Select the Type of Graph You Want To Visualize", ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"])

    # Define all the columns of the dataframe for the users to select the X-Axis and the Y-Axis as they please
    all_columns = df.columns
    
    # Line Chart
    if graph_type == "Line Chart":
        st.info(
            "For a Line Chart:"
            "\n\nx-axis: Typically represents a quantitative (numerical) variable, often indicating time or an ordered sequence, or an ordinal (ordered) "
            "categorical variable."
            "\n\ny-axis: Displays a quantitative (numerical) variable whose trend over the x-axis is being visualized."
        )
        try:
            # Configure the line chart
            choose_x_axis_line_chart = st.selectbox("Choose a Column for the X-Axis", all_columns)
            choose_y_axis_line_chart = st.selectbox("Choose a Column for the Y-Axis", all_columns)
            # Display the line chart
            if choose_x_axis_line_chart and choose_y_axis_line_chart:
                fig_line_chart = px.line(df, x=choose_x_axis_line_chart, y=choose_y_axis_line_chart)
                fig_line_chart.update_layout(
                    title=f"Line Chart of {choose_y_axis_line_chart} over {choose_x_axis_line_chart}",
                    xaxis_title=choose_x_axis_line_chart,
                    yaxis_title=choose_y_axis_line_chart,
                    hovermode="x unified",
                )
                st.plotly_chart(fig_line_chart)
                
        # Potential error handling
        except ValueError as ve:
            st.error(
                "Encountered a 'Value Error' during plotting the line chart!"
                f"\n\nDetails: {ve}"
            )
        except KeyError as ke:
            st.error(
                f"Encountered an 'Key Error': {ke}"
                "\n\nPlease check if the selected columns are correct."
            )
        except Exception as e:
            st.error(
                "An unexpected error occured during plotting the line chart!"
                f"\n\nDetails: {e}"
            )

    # Bar Chart
    elif graph_type == "Bar Chart":
        st.info(
            "For a Bar Chart:"
            "\n\nx-axis: Represents the categorical variable. Each category forms a base for a seperate bar."
            "\n\ny-axis: Represents the quantitative (numerical) variable. It shows the magnitude or frequency associated with each category on the x-axis."
        )
        try:
            # Configure the bar chart
            choose_x_axis_bar_chart = st.selectbox("Choose a Column for the X-Axis", all_columns)
            choose_y_axis_bar_chart = st.selectbox("Choose a Column for the Y-Axis", all_columns)
            # Display the bar chart
            if choose_x_axis_bar_chart and choose_y_axis_bar_chart:
                fig_bar_chart = px.bar(df, x=choose_x_axis_bar_chart, y=choose_y_axis_bar_chart)
                fig_bar_chart.update_layout(
                    title=f"Bar Chart of {choose_y_axis_bar_chart} by {choose_x_axis_bar_chart}",
                    xaxis_title=choose_x_axis_bar_chart,
                    yaxis_title=choose_y_axis_bar_chart,
                )
                st.plotly_chart(fig_bar_chart)

        # Potential error handling
        except ValueError as ve:
            st.error(
                "Encountered a 'Value Error' during plotting the bar chart!"
                f"\n\nDetails: {ve}"
            )
        except KeyError as ke:
            st.error(
                f"Encountered an 'Key Error': {ke}"
                "\n\nPlease check if the selected columns are correct."
            )
        except Exception as e:
            st.error(
                "An unexpected error occured during plotting the bar chart!"
                f"\n\nDetails: {e}"
            )

    # Scatter Plot
    elif graph_type == "Scatter Plot":
        st.info(
            "For a Scatter Plot:"
            "\n\nx-axis: Represents a quantitative (numerical) variable."
            "\n\ny-axis: Represents another quantitative (numerical) variable."
        )
        try:
            # Configure the scatter plot
            choose_x_axis_scatter_plot = st.selectbox("Choose a Column for the X-Axis", all_columns)
            choose_y_axis_scatter_plot = st.selectbox("Choose a Column for the Y-Axis", all_columns)
            # Display the scatter plot
            if choose_x_axis_scatter_plot and choose_y_axis_scatter_plot:
                fig_scatter_plot = px.scatter(
                    df,
                    x=choose_x_axis_scatter_plot,
                    y=choose_y_axis_scatter_plot,
                    title=f"Scatter Plot of {choose_y_axis_scatter_plot} vs. {choose_x_axis_scatter_plot}",
                    hover_data=all_columns)
                fig_scatter_plot.update_layout(
                    xaxis_title=choose_x_axis_scatter_plot,
                    yaxis_title=choose_y_axis_scatter_plot,
                )
                st.plotly_chart(fig_scatter_plot)

        # Potential error handling
        except ValueError as ve:
            st.error(
                "Encountered a 'Value Error' during plotting the scatter plot!"
                f"\n\nDetails: {ve}"
            )
        except KeyError as ke:
            st.error(
                f"Encountered an 'Key Error': {ke}"
                "\n\nPlease check if the selected columns are correct."
            )
        except Exception as e:
            st.error(
                "An unexpected error occured during plotting the scatter plot!"
                f"\n\nDetails: {e}"
            )

    # Histogram
    elif graph_type == "Histogram":
        st.info(
            "For a Histogram:"
            "\n\nx-axis: Designate a column containing a quantitative (numerical) variable, which is used to define the bins or intervals of the histogram."
            '\n\ny-axis: Inherently represents the "Frequency" or "Count" of the data points that fall within each defined bin of the x-axis variable'
        )
        try:
            # I. Data Preparation
            # Configure the histogram
            choose_numerical_col_histogram = st.selectbox("Choose a Numerical Column", df.select_dtypes(include=np.number).columns)
            
            # I.A. Handle the timedelta
            if pd.api.types.is_timedelta64_dtype(df[choose_numerical_col_histogram]):
                    
                # III.A. Widget for the "target_unit"
                unit_options = ["seconds", "minutes", "hours", "days", "months", "years"]
                target_unit = st.selectbox("Choose the Unit for Timedelta", unit_options, index=0)
                    
                df["histogram_value"] = standardize_timedelta(df, choose_numerical_col_histogram, target_unit=target_unit)
                plot_col = "histogram_value"

                # I.B. Aggregation to the nearest "X" units
                if pd.api.types.is_numeric_dtype(df[plot_col]):
                    
                    # III.B. Widgets for "decimals"
                    decimal = st.slider("Rounding Decimals:", min_value=0, max_value=5, value=2, key="decimal_rounding")
                    df["rounded_histogram_value"] = np.round(df["histogram_value"], decimals=decimal)

                    # Aggregated data
                    aggregated_data = df["rounded_histogram_value"].value_counts().reset_index()
                    aggregated_data.columns = ["value", "count"]

                    # III.C. Widget for the size of "data_sample"
                    sample_size = st.slider(
                        "Max Data Points to Plot:",
                        min_value=100,
                        max_value=max(100, int(len(aggregated_data)) if not aggregated_data.empty else 5000),
                        value=min(5000, int(len(aggregated_data)) if not aggregated_data.empty else 5000),
                        step=500,
                    )
                    st.info(
                        'The range and the initial value of the "Max Data Points to Plot" slider is dynamically determined by the number of data points in the '
                        "selected column."
                        "\n\nIf the slider appears not to move beyond a certain value, it indicates that the dataset currently being analyzed has a limited number of "
                        "data points."
                    )
                    # Sampling
                    if len(aggregated_data) > sample_size:
                        aggregated_data_sample = aggregated_data.sample(sample_size)
                    else:
                        aggregated_data_sample = aggregated_data

                    # III.D. Widget for "bins" - for timedelta
                    bin_number = st.slider("Number of Bins:", min_value=10, max_value=100, value=20)

                    # II.A. Plotting - for timedelta
                    # Plot the aggregated data
                    plt.figure(figsize=(10, 6))
                    sns.histplot(x=aggregated_data_sample["value"], kde=True, bins=bin_number, weights=aggregated_data_sample["count"])
                    plt.title(f"Histogram of {choose_numerical_col_histogram} (Aggregated)")
                    plt.xlabel(choose_numerical_col_histogram)
                    plt.ylabel("Frequency")
                    # IV.A. Display - for timedelta
                    st.pyplot(plt) 

            elif not pd.api.types.is_timedelta64_dtype(df[choose_numerical_col_histogram]):
                df["histogram_value"] = df[choose_numerical_col_histogram]
                plot_col = choose_numerical_col_histogram

                # III.E. Widget for "bins" - for column sother than timedelta
                bin_number_ottd = st.slider("Number of Bins:", min_value=10, max_value=100, value=20, key="number_of_bins_ottd")
                
                # II.A. Plotting - for columns other than timedelta
                # For columns other than timedelta
                plt.figure(figsize=(10, 6))
                sns.histplot(df[choose_numerical_col_histogram], kde=True, bins=bin_number_ottd)
                plt.title(f"Histogram of {choose_numerical_col_histogram}")
                plt.xlabel(choose_numerical_col_histogram)
                plt.ylabel("Frequency")
                # IV.A. Display - for columns other than timedelta
                st.pyplot(plt)

        # Potential error handling
        except ValueError as ve:
            st.error(
                "Encountered a 'Value Error' during plotting the histogram!"
                f"\n\nDetails: {ve}"
            )
        except KeyError as ke:
            st.error(
                f"Encountered an 'Key Error': {ke}"
                "\n\nPlease check if the selected columns are correct."
            )
        except Exception as e:
            st.error(
                "An unexpected error occured during plotting the histogram!"
                f"\n\nDetails: {e}"
            )

    # Box Plot
    elif graph_type == "Box Plot":
        st.info(
            "For a Box Plot:"
            "\n\ny-axis: Select a column representating a quantitative (numerical) variable, for which the distribution will be analyzed."
            "\n\nx-axis: Optionally select a categorical variable. If selected, the box plot will display the distribution of the quantitative (numerical) variable "
            "seperately for each category, thereby facilitating comparisions across groups."
        )
        try:
            # Configure the box plot
            choose_numerical_col_box_plot = st.selectbox("Choose a Numerical Column", df.select_dtypes(include=np.number).columns)
            # Display the box plot
            if choose_numerical_col_box_plot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=df[choose_numerical_col_box_plot])
                plt.title(f"Box Plot of {choose_numerical_col_box_plot}")
                plt.ylabel(choose_numerical_col_box_plot)
                st.pyplot(plt)

        # Potential error handling
        except ValueError as ve:
            st.error(
                "Encountered a 'Value Error' during plotting the box plot!"
                f"\n\nDetails: {ve}"
            )
        except KeyError as ke:
            st.error(
                f"Encountered an 'Key Error': {ke}"
                "\n\nPlease check if the selected columns are correct."
            )
        except Exception as e:
            st.error(
                "An unexpected error occured during plotting the box plot!"
                f"\n\nDetails: {e}"
            )

    # Pie Chart
    elif graph_type == "Pie Chart":
        st.info(
            "For a Pie Chart:"
            "\n\n- Designate a column containing a quantitative (numerical) variable. The values in this column will determine the size of each slice, "
            "typically representing the percentage or proportion of each category relative to the whole."
            "\n\n- Designate a column representing a categorical variable. The unique values in this column will define the slices of the pie chart, "
            "representing distinct categories."
        )
        try:
            # Configure the box plot
            choose_slice_pie_chart = st.selectbox("Choose a Numerical Column for Values", df.select_dtypes(include=np.number).columns)
            choose_color_pie_chart = st.selectbox("Choose a Categorical Column for Segments", df.select_dtypes(include=["object", "category"]).columns)
            # Display the pie chart
            if choose_slice_pie_chart and choose_color_pie_chart:
                fig_pie_chart = px.pie(df, values=choose_slice_pie_chart, names=choose_color_pie_chart, hole=0.3)
                fig_pie_chart.update_layout(
                    title=f"Pie Chart of {choose_slice_pie_chart} by {choose_color_pie_chart}",
                    showlegend=True
                )
                st.plotly_chart(fig_pie_chart)

        # Potential error handling
        except ValueError as ve:
            st.error(
                "Encountered a 'Value Error' during plotting the pie chart!"
                f"\n\nDetails: {ve}"
            )
        except KeyError as ke:
            st.error(
                f"Encountered an 'Key Error': {ke}"
                "\n\nPlease check if the selected columns are correct."
            )
        except Exception as e:
            st.error(
                "An unexpected error occured during plotting the pie chart!"
                f"\n\nDetails: {e}"
            )
