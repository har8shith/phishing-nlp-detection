from pathlib import Path
import pandas as pd
from config import DATA_PATH

def ensure_dataset() -> Path:
    data_path = DATA_PATH
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        return data_path
    rows = [
        {"subject": "Your Invoice for March", "body": "Hello, your invoice for March is attached. Please review it.", "label": 0},
        {"subject": "Meeting Reminder", "body": "Reminder: your meeting is scheduled for tomorrow at 3 PM.", "label": 0},
        {"subject": "Delivery Update", "body": "Your package has been dispatched and will arrive soon.", "label": 0},
        {"subject": "URGENT: Verify Your Account", "body": "Your account has been suspended. Click here: http://fake-verify-login.com", "label": 1},
        {"subject": "Security Alert", "body": "Unusual login detected. Secure at http://secure-login-alert.net", "label": 1},
        {"subject": "Payment Required Now", "body": "Your service will be terminated unless you pay now at http://billing-paynow-secure.com", "label": 1},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(data_path, index=False)
    return data_path
