"""
Script to combine all Excel files from the excels folder into a single CSV file.
Adds columns for the source Excel file name and sheet name.
"""

import pandas as pd
from pathlib import Path


def combine_excel_files(excels_folder: str = "excels", output_file: str = "combined_data.csv"):
    """
    Reads all Excel files from the specified folder, combines all sheets
    from all files into a single DataFrame, and saves it as a CSV.
    
    Args:
        excels_folder: Path to the folder containing Excel files
        output_file: Name of the output CSV file
    """
    excels_path = Path(excels_folder)
    
    if not excels_path.exists():
        print(f"Error: Folder '{excels_folder}' does not exist.")
        return
    
    # Find all Excel files
    excel_files = list(excels_path.glob("*.xlsx")) + list(excels_path.glob("*.xls"))
    
    if not excel_files:
        print(f"No Excel files found in '{excels_folder}'.")
        return
    
    print(f"Found {len(excel_files)} Excel file(s):")
    for f in excel_files:
        print(f"  - {f.name}")
    
    all_dataframes = []
    
    for excel_file in excel_files:
        print(f"\nProcessing: {excel_file.name}")
        
        # Read all sheets from the Excel file
        excel_data = pd.ExcelFile(excel_file)
        sheet_names = excel_data.sheet_names
        
        print(f"  Sheets found: {sheet_names}")
        
        for sheet_name in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Add columns for file name and sheet name
            df.insert(0, "excel_file", excel_file.name)
            df.insert(1, "sheet_name", sheet_name)
            
            all_dataframes.append(df)
            print(f"    - '{sheet_name}': {len(df)} rows")
    
    if not all_dataframes:
        print("\nNo data found in any Excel file.")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*50}")
    print(f"Combined data saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
    print(f"Columns: {list(combined_df.columns)}")


if __name__ == "__main__":
    combine_excel_files()

