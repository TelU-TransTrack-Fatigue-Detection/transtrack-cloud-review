#!/usr/bin/env python3
"""Manual inference test - test the full pipeline with real/sample data"""

import asyncio
import json
from pathlib import Path
from app.inference import run_inference
from app.pipeline import predict as pipeline_predict

async def test_inference_pipeline():
    """Test the complete inference pipeline"""
    
    # Check if test video exists
    test_video_path = Path("test_video.mp4")
    if not test_video_path.exists():
        print(f"❌ Test video not found: {test_video_path}")
        print("   Please provide a test_video.mp4 file")
        return
    
    print(f"✅ Found test video: {test_video_path}")
    
    try:
        # Test 1: Run inference
        print("\n📹 Testing inference pipeline...")
        result = await run_inference(str(test_video_path), "eyes_closed")
        
        print(f"\n✅ Inference completed successfully!")
        print(f"   Result: {json.dumps(result, indent=2)}")
        
        # Check required keys
        required_keys = ["video_url_after_process", "confidence_level", "review_result", "other"]
        missing_keys = [k for k in required_keys if k not in result]
        
        if missing_keys:
            print(f"\n❌ Missing keys: {missing_keys}")
        else:
            print(f"\n✅ All required keys present")
        
        # Validate values
        print(f"\n📊 Validation:")
        print(f"   Confidence Level: {result.get('confidence_level')}% (0-100)")
        print(f"   Review Result: {result.get('review_result')} (bool)")
        print(f"   Label: {result.get('other', {}).get('label', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_inference_pipeline())
    exit(0 if success else 1)
