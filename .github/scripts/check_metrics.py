import json
import sys
from pathlib import Path

# Baseline target based on your notebook evaluation 
BASELINE_F1 = 0.6900
METRICS_PATH = Path("reports/metrics.json")

def main():        
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    f1_macro = metrics.get("f1_macro", 0.0)
    print(f"📈 Baseline Target : {BASELINE_F1:.4f}")
    print(f"📊 Current F1 Macro: {f1_macro:.4f}")
    
    if f1_macro < BASELINE_F1:
        print("❌ FATAL: Model quality has degraded below the strict baseline!")
        sys.exit(1) # This exit code tells GitHub Actions to highlight the phase in RED
        
    print("✅ Model meets baseline quality standard!")

if __name__ == "__main__":
    main()
