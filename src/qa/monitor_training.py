"""Monitor training progress"""
import json
from pathlib import Path
import time

log_file = Path("outputs/qa_model/trainer_state.json")

print("Monitoring training progress...\n")

while True:
    if log_file.exists():
        with open(log_file) as f:
            state = json.load(f)
        
        if 'log_history' in state:
            latest = state['log_history'][-1]
            print(f"Step {latest.get('step', 0):5d} | "
                  f"Loss: {latest.get('loss', 0):.4f} | "
                  f"Eval Loss: {latest.get('eval_loss', 'N/A')}")
    
    time.sleep(10)
