import os
import pandas as pd
import numpy as np

def detect_naming_issues(df):
    """Detects trailing spaces, casing differences, and potential typos."""
    issues = []
    columns = df.columns.tolist()
    
    # Check for trailing/leading spaces
    for col in columns:
        if col != col.strip():
            issues.append(f"Column '{col}' has trailing or leading spaces.")
            
    # Check for casing differences (e.g., 'Date' and 'date')
    lower_cols = [c.lower().strip() for c in columns]
    if len(set(lower_cols)) < len(lower_cols):
        seen = set()
        duplicates = set()
        for c in lower_cols:
            if c in seen:
                duplicates.add(c)
            seen.add(c)
        issues.append(f"Potential casing/case-insensitive duplicates: {list(duplicates)}")
        
    # Check for typos (specific ones mentioned: fribal vs friability)
    for col in lower_cols:
        if 'fribal' in col and 'friability' not in col:
            issues.append(f"Potential typo in column '{col}' (could be 'friability').")
            
    return issues

def inspect_file(file_path, file_type=""):
    """Inspects an Excel file and returns a summary dictionary."""
    summary = {
        "file_name": os.path.basename(file_path),
        "sheet_names": [],
        "total_sheets": 0,
        "sample_sheets": [],
        "shape": None,
        "columns": [],
        "missing_summary": {},
        "issues": [],
        "missing_20_plus": [],
        "data_types": None,
        "head": None
    }
    
    try:
        # Load excel file to get sheet names
        xl = pd.ExcelFile(file_path)
        summary["sheet_names"] = xl.sheet_names
        summary["total_sheets"] = len(xl.sheet_names)
        summary["sample_sheets"] = xl.sheet_names[:5]
        
        # Load first sheet
        df = xl.parse(xl.sheet_names[0])
        summary["shape"] = df.shape
        summary["columns"] = df.columns.tolist()
        summary["missing_summary"] = df.isnull().sum().to_dict()
        summary["data_types"] = df.dtypes.to_dict()
        summary["head"] = df.head(5)
        
        # Check for column with >20% missing values
        missing_percent = (df.isnull().sum() / len(df)) * 100
        summary["missing_20_plus"] = missing_percent[missing_percent > 20].index.tolist()
        
        # Detect naming issues
        summary["issues"] = detect_naming_issues(df)
        
        # Check for duplicate columns
        if len(set(df.columns)) < len(df.columns):
            summary["issues"].append("Duplicate column names detected.")
            
    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")
        return None
        
    return summary

def main():
    # Setup paths
    # Assuming script is in 'code' folder, dataset is in 'dataset' folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(base_dir), "dataset")
    output_dir = os.path.join(os.path.dirname(base_dir), "outputs")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at {output_dir}")
        
    # Detect files
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} not found.")
        return
        
    files = os.listdir(dataset_dir)
    print(f"Files detected in {dataset_dir}: {files}")
    
    # Identify production and process files
    production_file = None
    process_file = None
    
    for f in files:
        if "production" in f.lower() and f.endswith(".xlsx"):
            production_file = os.path.join(dataset_dir, f)
        elif "process" in f.lower() and f.endswith(".xlsx"):
            process_file = os.path.join(dataset_dir, f)
            
    report_content = "===============================\nDATASET INSPECTION REPORT\n===============================\n\n"
    
    # Final Summary for Terminal
    print("\nStarting dataset inspection...")
    
    # 1. Process Production File
    if production_file:
        print(f"\nInspecting Production File: {os.path.basename(production_file)}")
        prod_summary = inspect_file(production_file)
        if prod_summary:
            print(f"Sheet names: {prod_summary['sheet_names']}")
            print(f"Shape: {prod_summary['shape']}")
            print(f"Columns: {prod_summary['columns']}")
            print("First 5 rows:")
            print(prod_summary['head'])
            print("\nData Types:")
            print(prod_summary['data_types'])
            print("\nMissing Values Count:")
            print(prod_summary['missing_summary'])
            
            report_content += f"Production File:\n"
            report_content += f"- File name: {prod_summary['file_name']}\n"
            report_content += f"- Sheet names: {prod_summary['sheet_names']}\n"
            report_content += f"- Shape: {prod_summary['shape']}\n"
            report_content += f"- Columns: {prod_summary['columns']}\n"
            report_content += f"- Missing summary: {prod_summary['missing_summary']}\n\n"
        else:
            report_content += "Production File: Error loading or file not found.\n\n"
    else:
        print("\nProduction file not detected.")
        report_content += "Production File: Not found.\n\n"
        prod_summary = None

    # 2. Process File
    if process_file:
        print(f"\nInspecting Process File: {os.path.basename(process_file)}")
        proc_summary = inspect_file(process_file)
        if proc_summary:
            print(f"Total number of sheets: {proc_summary['total_sheets']}")
            print(f"First 5 sheet names: {proc_summary['sample_sheets']}")
            print(f"Shape: {proc_summary['shape']}")
            print(f"Columns: {proc_summary['columns']}")
            print("First 5 rows:")
            print(proc_summary['head'])
            print("\nData Types:")
            print(proc_summary['data_types'])
            print("\nMissing Values Count:")
            print(proc_summary['missing_summary'])
            
            report_content += f"Process File:\n"
            report_content += f"- File name: {proc_summary['file_name']}\n"
            report_content += f"- Total sheets: {proc_summary['total_sheets']}\n"
            report_content += f"- Sample sheet name: {proc_summary['sample_sheets'][0] if proc_summary['sample_sheets'] else 'N/A'}\n"
            report_content += f"- Shape: {proc_summary['shape']}\n"
            report_content += f"- Columns: {proc_summary['columns']}\n"
            report_content += f"- Missing summary: {proc_summary['missing_summary']}\n\n"
        else:
            report_content += "Process File: Error loading or file not found.\n\n"
    else:
        print("\nProcess file not detected.")
        report_content += "Process File: Not found.\n\n"
        proc_summary = None

    # Potential Issues Section
    report_content += "Potential Issues Detected:\n"
    potential_issues_list = []
    
    if prod_summary:
        potential_issues_list.extend([f"[Production] {issue}" for issue in prod_summary['issues']])
        prod_missing_20 = prod_summary['missing_20_plus']
        if prod_missing_20:
            potential_issues_list.append(f"[Production] Columns with >20% missing values: {prod_missing_20}")
            
    if proc_summary:
        potential_issues_list.extend([f"[Process] {issue}" for issue in proc_summary['issues']])
        proc_missing_20 = proc_summary['missing_20_plus']
        if proc_missing_20:
            potential_issues_list.append(f"[Process] Columns with >20% missing values: {proc_missing_20}")
            
    if not potential_issues_list:
        report_content += "- None detected.\n"
    else:
        for issue in potential_issues_list:
            report_content += f"- {issue}\n"
            
    # Save Report
    report_path = os.path.join(output_dir, "dataset_inspection_report.txt")
    try:
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"\nReport successfully generated at: {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}")

    print("\n===============================")
    print("FINAL SUMMARY")
    print("===============================")
    print(f"Production File detected: {'Yes' if production_file else 'No'}")
    print(f"Process File detected: {'Yes' if process_file else 'No'}")
    print(f"Report saved: {'Yes' if os.path.exists(report_path) else 'No'}")
    print("===============================\n")

if __name__ == "__main__":
    main()
