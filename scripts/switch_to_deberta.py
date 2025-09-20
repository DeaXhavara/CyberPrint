#!/usr/bin/env python3
"""
Switch CyberPrint Pipeline to Use Trained DeBERTa Model

This script updates the CyberPrint pipeline to use the newly trained DeBERTa model
instead of the default logistic regression model.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

def backup_current_predictor():
    """Backup the current ML predictor"""
    predictor_path = Path(__file__).parent.parent / "cyberprint_ml_predictor.py"
    backup_path = predictor_path.with_suffix('.py.backup')
    
    if predictor_path.exists():
        shutil.copy2(predictor_path, backup_path)
        print(f"✅ Backed up current predictor to {backup_path}")
        return True
    return False

def update_pipeline_config(model_dir):
    """Update the pipeline to use DeBERTa predictor"""
    pipeline_path = Path(__file__).parent.parent / "cyberprint" / "pipeline.py"
    
    if not pipeline_path.exists():
        print(f"❌ Pipeline file not found: {pipeline_path}")
        return False
    
    # Read current pipeline
    with open(pipeline_path, 'r') as f:
        content = f.read()
    
    # Check if already using DeBERTa
    if 'DeBERTaCyberPrintPredictor' in content:
        print("ℹ️  Pipeline already configured for DeBERTa")
        return True
    
    # Add DeBERTa import
    import_line = "from .models.ml.deberta_predictor import DeBERTaCyberPrintPredictor"
    
    if import_line not in content:
        # Find the imports section and add our import
        lines = content.split('\n')
        import_inserted = False
        
        for i, line in enumerate(lines):
            if line.startswith('from .models.') or line.startswith('from cyberprint.models.'):
                lines.insert(i + 1, import_line)
                import_inserted = True
                break
        
        if not import_inserted:
            # Find any import and add after it
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    lines.insert(i + 1, import_line)
                    break
        
        content = '\n'.join(lines)
    
    # Replace ML predictor initialization
    old_init = "self.ml_predictor = CyberPrintMLPredictor()"
    new_init = f'self.ml_predictor = DeBERTaCyberPrintPredictor("{model_dir}", fallback_to_logistic=True)'
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        print("✅ Updated ML predictor initialization")
    else:
        print("⚠️  Could not find ML predictor initialization to replace")
        return False
    
    # Write updated pipeline
    with open(pipeline_path, 'w') as f:
        f.write(content)
    
    print("✅ Pipeline updated to use DeBERTa predictor")
    return True

def create_test_script(model_dir):
    """Create a test script to verify the integration"""
    test_script = f'''#!/usr/bin/env python3
"""
Test DeBERTa Integration with CyberPrint Pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cyberprint.pipeline import CyberPrintPipeline

def test_deberta_integration():
    """Test that DeBERTa integration works"""
    print("🧪 Testing DeBERTa Integration")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = CyberPrintPipeline()
    
    # Test examples
    test_texts = [
        "Thank you so much for your help!",
        "This is absolutely terrible and useless.",
        "The algorithm works as expected in most cases.",
        "LOL this is hilarious! 😂"
    ]
    
    print("Testing predictions...")
    for i, text in enumerate(test_texts, 1):
        try:
            result = pipeline.ml_predictor.predict_text([text])[0]
            print(f"{{i}}. Text: {{text[:50]}}...")
            print(f"   Sentiment: {{result['sentiment']}} ({{result['confidence']:.3f}})")
            if 'sub_label' in result:
                print(f"   Sub-label: {{result['sub_label']}}")
            print(f"   Model: {{result.get('model_type', 'unknown')}}")
            print()
        except Exception as e:
            print(f"❌ Error with text {{i}}: {{e}}")
            return False
    
    # Test model info
    try:
        info = pipeline.ml_predictor.get_model_info()
        print("Model Information:")
        for key, value in info.items():
            print(f"  {{key}}: {{value}}")
    except Exception as e:
        print(f"⚠️  Could not get model info: {{e}}")
    
    print("✅ DeBERTa integration test completed successfully!")
    return True

if __name__ == "__main__":
    test_deberta_integration()
'''
    
    test_path = Path(__file__).parent / "test_deberta_integration.py"
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    # Make executable
    os.chmod(test_path, 0o755)
    print(f"✅ Created test script: {test_path}")

def main():
    parser = argparse.ArgumentParser(description="Switch CyberPrint to use trained DeBERTa model")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained DeBERTa model')
    parser.add_argument('--backup', action='store_true', help='Backup current predictor before switching')
    parser.add_argument('--test', action='store_true', help='Create test script')
    args = parser.parse_args()
    
    print("🔄 Switching CyberPrint to DeBERTa Model")
    print("=" * 50)
    
    # Verify model directory exists
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return 1
    
    # Check for required files
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    
    if missing_files:
        print(f"⚠️  Some model files may be missing: {missing_files}")
        print("Continuing anyway...")
    
    # Backup if requested
    if args.backup:
        backup_current_predictor()
    
    # Update pipeline
    success = update_pipeline_config(str(model_path.absolute()))
    
    if not success:
        print("❌ Failed to update pipeline configuration")
        return 1
    
    # Create test script if requested
    if args.test:
        create_test_script(str(model_path.absolute()))
    
    print("\n" + "=" * 50)
    print("✅ Successfully switched CyberPrint to DeBERTa model!")
    print(f"📁 Model directory: {model_path.absolute()}")
    print("\nNext steps:")
    print("1. Test the integration with: python scripts/test_deberta_integration.py")
    print("2. Run your CyberPrint pipeline as usual")
    print("3. Monitor performance and accuracy improvements")
    
    if args.backup:
        print("4. Remove backup file if everything works well")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
