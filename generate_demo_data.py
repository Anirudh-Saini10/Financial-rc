import pandas as pd
import os

# Sheet A: Internal Ledger
sheet_a_data = [
    {"Date": "2023-10-01", "Transaction ID": "TXN-001", "Vendor Name": "TechCorp Solutions", "Amount": -1500.00, "Type": "Debit"},
    {"Date": "2023-10-02", "Transaction ID": "TXN-002", "Vendor Name": "Office Supplies Inc", "Amount": -250.50, "Type": "Debit"},
    {"Date": "2023-10-03", "Transaction ID": "TXN-003", "Vendor Name": "Client A", "Amount": 5000.00, "Type": "Credit"},
    {"Date": "2023-10-04", "Transaction ID": "TXN-004", "Vendor Name": "Software Subscriptions", "Amount": -99.99, "Type": "Debit"},
    {"Date": "2023-10-05", "Transaction ID": "TXN-005", "Vendor Name": "Consulting Fee", "Amount": -1200.00, "Type": "Debit"},
    {"Date": "2023-10-06", "Transaction ID": "TXN-006", "Vendor Name": "Client B", "Amount": 8500.00, "Type": "Credit"},
    {"Date": "2023-10-07", "Transaction ID": "TXN-007", "Vendor Name": "Marketing Ads", "Amount": -500.00, "Type": "Debit"},
    {"Date": "2023-10-08", "Transaction ID": "TXN-008", "Vendor Name": "Server Hosting", "Amount": -300.00, "Type": "Debit"},
    {"Date": "2023-10-09", "Transaction ID": "TXN-009", "Vendor Name": "Suspicious Huge Transfer", "Amount": -950000.00, "Type": "Debit"}, # Anomaly
]

# Sheet B: External Statement (Bank/Vendor)
sheet_b_data = [
    {"Date": "2023-10-01", "Reference": "TXN-001", "Counterparty": "TechCorp Solutions LLC", "Value": -1500.00, "Direction": "Outflow"}, # Exact match (fuzzy name)
    {"Date": "2023-10-02", "Reference": "TXN-002", "Counterparty": "Office Supplies Inc", "Value": -250.50, "Direction": "Outflow"}, # Exact match
    {"Date": "2023-10-03", "Reference": "TXN-003", "Counterparty": "Client A", "Value": 5000.00, "Direction": "Inflow"}, # Exact match
    {"Date": "2023-10-04", "Reference": "TXN-004", "Counterparty": "Software Subs", "Value": -109.99, "Direction": "Outflow"}, # Mismatched amount
    {"Date": "2023-10-06", "Reference": "TXN-006", "Counterparty": "Client B", "Value": 8500.00, "Direction": "Inflow"}, # Exact match
    {"Date": "2023-10-07", "Reference": "TXN-007", "Counterparty": "Marketing Ads", "Value": -500.00, "Direction": "Outflow"}, # Exact match
    {"Date": "2023-10-08", "Reference": "TXN-008", "Counterparty": "Server Hosting", "Value": -300.00, "Direction": "Outflow"}, # Exact match
    {"Date": "2023-10-10", "Reference": "BANK-FEE", "Counterparty": "Monthly Bank Fee", "Value": -15.00, "Direction": "Outflow"}, # Missing in A
    {"Date": "2023-10-09", "Reference": "TXN-009", "Counterparty": "Unknown Transfer", "Value": -950000.00, "Direction": "Outflow"}, # Match for anomaly
]

os.makedirs('sample_data', exist_ok=True)
pd.DataFrame(sheet_a_data).to_excel('sample_data/demo_sheet_a.xlsx', index=False)
pd.DataFrame(sheet_b_data).to_excel('sample_data/demo_sheet_b.xlsx', index=False)
print("Demo sheets created successfully.")
