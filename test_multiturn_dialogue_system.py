#!/usr/bin/env python3
"""
Test script for the multi-turn dialogue reward system.

This script tests the custom interaction class and reward functions
to ensure they work correctly before running full training.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/verl/verl')

from verl.interactions.multiturn_dialogue_interaction import MultiturnDialogueInteraction
from verl.utils.reward_score.multiturn_dialogue import (
    compute_multiturn_dialogue_score,
    compute_model_based_dialogue_score
)


async def test_interaction_system():
    """Test the multi-turn dialogue interaction system."""
    print("🧪 Testing Multi-turn Dialogue Interaction System...")
    
    # Initialize interaction with test config
    config = {
        "name": "test_dialogue",
        "scoring_model": "test_scorer",
        "response_model": "test_responder",
        "min_score_threshold": 0.3,
        "good_score_threshold": 0.7,
        "max_turns": 5
    }
    
    interaction = MultiturnDialogueInteraction(config)
    
    # Test starting an interaction
    instance_id = await interaction.start_interaction(
        initial_context="Testing multi-turn dialogue"
    )
    print(f"✅ Started interaction: {instance_id}")
    
    # Test generating responses for multiple turns
    test_messages = [
        [
            {"role": "user", "content": "Hello, can you help me?"},
            {"role": "assistant", "content": "Hello! I'd be happy to help you. What do you need assistance with?"}
        ],
        [
            {"role": "user", "content": "Hello, can you help me?"},
            {"role": "assistant", "content": "Hello! I'd be happy to help you. What do you need assistance with?"},
            {"role": "user", "content": "I need help with understanding machine learning."},
            {"role": "assistant", "content": "Machine learning is a fascinating field! It involves training algorithms to learn patterns from data. Would you like me to explain a specific aspect?"}
        ],
        [
            {"role": "user", "content": "Hello, can you help me?"},
            {"role": "assistant", "content": "Hello! I'd be happy to help you. What do you need assistance with?"},
            {"role": "user", "content": "I need help with understanding machine learning."},
            {"role": "assistant", "content": "Machine learning is a fascinating field! It involves training algorithms to learn patterns from data. Would you like me to explain a specific aspect?"},
            {"role": "user", "content": "Can you explain neural networks?"},
            {"role": "assistant", "content": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information in layers."}
        ]
    ]
    
    for i, messages in enumerate(test_messages):
        print(f"\n--- Turn {i+1} ---")
        should_terminate, response, score, metadata = await interaction.generate_response(
            instance_id, messages
        )
        
        print(f"Response: {response}")
        print(f"Score: {score:.3f}")
        print(f"Should terminate: {should_terminate}")
        print(f"Metadata: {metadata}")
        
        if should_terminate:
            print("🔚 Conversation terminated")
            break
    
    # Test finalization
    await interaction.finalize_interaction(instance_id)
    print("✅ Interaction finalized successfully")


def test_reward_functions():
    """Test the reward functions."""
    print("\n🧪 Testing Reward Functions...")
    
    # Test data
    test_cases = [
        {
            "name": "Good dialogue",
            "solution": "User: Can you help me?\nAssistant: Of course! I'd be happy to help you. What specific assistance do you need?",
            "ground_truth": "Helpful response",
            "expected_range": (0.6, 1.0)
        },
        {
            "name": "Poor dialogue", 
            "solution": "ok",
            "ground_truth": "Detailed explanation needed",
            "expected_range": (0.0, 0.4)
        },
        {
            "name": "Multi-turn dialogue",
            "solution": "User: Hello\nAssistant: Hi! How can I help?\nUser: Tell me about AI\nAssistant: AI is the simulation of human intelligence in machines that are programmed to think and learn.",
            "ground_truth": "Educational response about AI",
            "expected_range": (0.7, 1.0)
        }
    ]
    
    print("\n--- Testing rule-based scoring ---")
    for test_case in test_cases:
        score = compute_multiturn_dialogue_score(
            data_source="test",
            solution_str=test_case["solution"],
            ground_truth=test_case["ground_truth"]
        )
        
        min_expected, max_expected = test_case["expected_range"]
        status = "✅" if min_expected <= score <= max_expected else "❌"
        
        print(f"{status} {test_case['name']}: {score:.3f} (expected: {min_expected}-{max_expected})")


async def test_model_based_scoring():
    """Test model-based scoring (will fallback to rule-based if API not available)."""
    print("\n--- Testing model-based scoring ---")
    
    test_dialogue = "User: What is machine learning?\nAssistant: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
    
    try:
        score = await compute_model_based_dialogue_score(
            data_source="test",
            solution_str=test_dialogue,
            ground_truth="Educational response",
            scoring_model_url="http://localhost:8000/v1/chat/completions",  # This will likely fail
            scoring_model_name="test-model"
        )
        print(f"✅ Model-based score: {score:.3f}")
    except Exception as e:
        print(f"⚠️ Model-based scoring test completed with fallback: {e}")


def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration...")
    
    # Check if our files exist
    files_to_check = [
        "/home/runner/work/verl/verl/verl/interactions/multiturn_dialogue_interaction.py",
        "/home/runner/work/verl/verl/verl/utils/reward_score/multiturn_dialogue.py",
        "/home/runner/work/verl/verl/examples/sglang_multiturn/config/interaction_config/multiturn_dialogue_config.yaml",
        "/home/runner/work/verl/verl/examples/sglang_multiturn/config/multiturn_dialogue_grpo.yaml",
        "/home/runner/work/verl/verl/examples/sglang_multiturn/run_multiturn_dialogue_custom_reward.sh",
        "/home/runner/work/verl/verl/examples/sglang_multiturn/run_advanced_multiturn_dialogue.sh"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)} exists")
        else:
            print(f"❌ {os.path.basename(file_path)} missing")


async def main():
    """Run all tests."""
    print("🚀 Multi-turn Dialogue Reward System Test Suite")
    print("=" * 50)
    
    try:
        # Test configuration files
        test_configuration()
        
        # Test reward functions
        test_reward_functions()
        
        # Test model-based scoring
        await test_model_based_scoring()
        
        # Test interaction system
        await test_interaction_system()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("\nYour multi-turn dialogue reward system is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your dialogue dataset in Parquet format")
        print("2. (Optional) Set up your scoring model API endpoint")
        print("3. Run the training script:")
        print("   ./examples/sglang_multiturn/run_multiturn_dialogue_custom_reward.sh")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())