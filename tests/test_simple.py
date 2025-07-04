"""
Simple test to verify workflow components work.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    # Test basic imports
    from flow import (
        NodeConfig,
        DataFlowConfig,
        FallbackStrategy,
        CircuitBreakerConfig,
        CircuitBreakerState,
        NodeExecutionMode,
        WorkflowError,
        ErrorSeverity
    )
    
    print("✓ Successfully imported configuration classes")
    
    # Test NodeConfig creation
    config = NodeConfig(name="TestNode")
    assert config.name == "TestNode"
    assert config.enabled == True
    assert config.required == True
    print("✓ NodeConfig creation works")
    
    # Test DataFlowConfig creation
    data_config = DataFlowConfig()
    assert data_config.input_mapping == {}
    print("✓ DataFlowConfig creation works")
    
    # Test FallbackStrategy creation
    fallback = FallbackStrategy()
    assert fallback.enable_transcript_only == True
    print("✓ FallbackStrategy creation works")
    
    # Test CircuitBreakerConfig creation
    cb_config = CircuitBreakerConfig()
    assert cb_config.failure_threshold == 3
    print("✓ CircuitBreakerConfig creation works")
    
    # Test WorkflowError creation
    error = WorkflowError(
        flow_name="test",
        error_type="ValueError",
        message="test message"
    )
    assert error.flow_name == "test"
    print("✓ WorkflowError creation works")
    
    # Test ErrorSeverity enum
    assert ErrorSeverity.HIGH.value == "high"
    print("✓ ErrorSeverity enum works")
    
    print("\n🎉 All basic workflow component tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)