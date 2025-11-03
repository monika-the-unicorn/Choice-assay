"""
Quick test to verify fleet configuration loads correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_fleet_config():
    """Test that fleet config can be imported and devices created"""
    try:
        # Add the choice_assay directory to the path
        choice_assay_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'choice_assay')
        sys.path.insert(0, choice_assay_path)
        
        from my_fleet_config import INVENTORY, create_choice_assay_device
        
        print("✓ Fleet config imported successfully")
        
        # Test device creation
        dp_trees = create_choice_assay_device()
        print(f"✓ Device creation successful, created {len(dp_trees)} DP trees")
        
        # Test inventory
        print(f"✓ Fleet inventory contains {len(INVENTORY)} devices")
        
        for device in INVENTORY:
            print(f"  Device: {device.name} - {device.notes}")
            print(f"    ID: {device.device_id}")
            print(f"    Tags: {device.tags}")
        
        print("\n🎉 Fleet configuration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Fleet configuration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fleet_config()
    exit(0 if success else 1)