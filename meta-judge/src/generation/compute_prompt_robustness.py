import statistics
from collections import defaultdict

def analyze_prompt_robustness(filepath):
    # Dictionary to hold the metrics for each model configuration
    results = defaultdict(lambda: {'Spearman': [], 'Kendall': []})
    
    # Read the markdown file
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines or headers (lines starting with ##)
            if not line or line.startswith('#'):
                continue
            
            # Split the line by the pipe character '|'
            parts = [p.strip() for p in line.split('|')]
            
            if len(parts) >= 4:
                model_config = parts[0]
                model_config = '_'.join(model_config.split('_')[1:-2])
                
                # Extract the primary metric (the float before the parenthesis)
                # e.g., "0.8960   (0.0000)" -> "0.8960" -> 0.8960
                try:
                    Spearman = float(parts[2].split()[0])
                    Kendall = float(parts[3].split()[0])
                    
                    results[model_config]['Spearman'].append(Spearman)
                    results[model_config]['Kendall'].append(Kendall)
                except ValueError:
                    # Skip lines that don't have properly formatted numbers
                    continue

    # Print the aggregated results
    print(f"{'Model Configuration':<60} | {'Spearman (Mean ± Std)':<22} | {'Kendall (Mean ± Std)':<22}")
    print("-" * 110)
    
    for model, metrics in results.items():
        m1_vals = metrics['Spearman']
        m2_vals = metrics['Kendall']
        
        # Calculate mean
        m1_mean = statistics.mean(m1_vals)
        m2_mean = statistics.mean(m2_vals)
        
        # Calculate sample standard deviation (needs at least 2 data points)
        m1_std = statistics.stdev(m1_vals) if len(m1_vals) > 1 else 0.0
        m2_std = statistics.stdev(m2_vals) if len(m2_vals) > 1 else 0.0
        
        print(f"{model:<60} | {m1_mean:.4f} ± {m1_std:.4f}   | {m2_mean:.4f} ± {m2_std:.4f}")

if __name__ == "__main__":
    analyze_prompt_robustness('/mnt/proj1/fta-25-74/dev/master_thesis/src/karolina_testing/testing_sets_metacorr.md')
    pass