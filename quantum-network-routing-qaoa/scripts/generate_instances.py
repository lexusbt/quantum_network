"""
Generate all 200 routing problem instances
Run with: python -m scripts.generate_instances
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qubo_generation.instance_generator import InstanceGenerator
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/instance_generation.log'),
        logging.StreamHandler()
    ]
)

def main():
    print("\n" + "="*70)
    print("QUANTUM NETWORK ROUTING - INSTANCE GENERATION")
    print("="*70)
    print("\nThis will generate 200 routing problem instances")
    print("Estimated time: 30-90 minutes depending on your machine")
    print("\nPress Ctrl+C at any time to stop (progress will be saved)")
    print("="*70 + "\n")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        generator = InstanceGenerator()
        instances = generator.generate_all_instances(n_instances=200, seed=42)
        
        print("\n" + "="*70)
        print("✓ INSTANCE GENERATION COMPLETE!")
        print("="*70)
        print(f"\nGenerated: {len(instances)} instances")
        print(f"Location: {generator.instances_dir}")
        print(f"Train: {len([i for i in instances if i['split'] == 'train'])}")
        print(f"Val: {len([i for i in instances if i['split'] == 'validation'])}")
        print(f"Test: {len([i for i in instances if i['split'] == 'test'])}")
        print(f"\nMaster index: {generator.instances_dir / 'master_index.csv'}")
        print(f"Log file: logs/instance_generation.log")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        print("Partial progress has been saved")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()