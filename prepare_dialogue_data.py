#!/usr/bin/env python3
"""
Sample data preparation script for multi-turn dialogue training.

This script demonstrates how to prepare your dialogue data for training
with the verl multi-turn dialogue reward system.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def create_sample_dialogue_data() -> List[Dict[str, Any]]:
    """Create sample multi-turn dialogue data for demonstration."""
    
    sample_dialogues = [
        {
            "messages": [
                {"role": "user", "content": "Can you help me understand machine learning?"},
                {"role": "assistant", "content": "I'd be happy to help you understand machine learning! Machine learning is a subset of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, rather than being explicitly programmed for every task."},
                {"role": "user", "content": "What are the main types of machine learning?"},
                {"role": "assistant", "content": "There are three main types: 1) Supervised learning - learning from labeled examples, 2) Unsupervised learning - finding patterns in unlabeled data, and 3) Reinforcement learning - learning through trial and error with rewards and penalties."},
                {"role": "user", "content": "Can you give me an example of supervised learning?"},
                {"role": "assistant", "content": "Sure! A classic example is email spam detection. You train a model using thousands of emails that are already labeled as 'spam' or 'not spam'. The model learns to identify patterns in spam emails (like certain words or phrases) and can then classify new emails automatically."}
            ],
            "data_source": "educational_dialogue",
            "ground_truth": "Educational explanation about machine learning concepts"
        },
        {
            "messages": [
                {"role": "user", "content": "I'm feeling stressed about my upcoming presentation."},
                {"role": "assistant", "content": "I understand that presentations can feel overwhelming. It's completely normal to feel stressed about them. Would you like some strategies to help manage that stress and prepare effectively?"},
                {"role": "user", "content": "Yes, that would be helpful. What should I do?"},
                {"role": "assistant", "content": "Here are some effective strategies: 1) Practice your presentation multiple times out loud, 2) Prepare for potential questions, 3) Use breathing exercises to calm nerves, 4) Visualize success, and 5) Arrive early to test equipment. Which of these would you like me to elaborate on?"},
                {"role": "user", "content": "Can you tell me more about breathing exercises?"},
                {"role": "assistant", "content": "Absolutely! Try the 4-7-8 technique: breathe in for 4 counts, hold for 7 counts, then exhale for 8 counts. This activates your parasympathetic nervous system and helps reduce anxiety. You can do this a few times before your presentation to center yourself."}
            ],
            "data_source": "support_dialogue", 
            "ground_truth": "Supportive guidance for presentation anxiety"
        },
        {
            "messages": [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to real-time weather data, but I can suggest some ways for you to check the weather: you could use a weather app, check a weather website like weather.com, or ask a voice assistant with internet access."},
                {"role": "user", "content": "OK"},
                {"role": "assistant", "content": "Is there anything else I can help you with today? Perhaps I could assist with planning activities based on different weather conditions, or help you with something else entirely?"}
            ],
            "data_source": "general_dialogue",
            "ground_truth": "Helpful response about weather information"
        },
        {
            "messages": [
                {"role": "user", "content": "How do I cook pasta?"},
                {"role": "assistant", "content": "Here's how to cook pasta perfectly: 1) Boil a large pot of salted water, 2) Add pasta when water is at a rolling boil, 3) Stir occasionally to prevent sticking, 4) Cook according to package directions (usually 8-12 minutes), 5) Test for doneness - it should be al dente (firm but not hard), 6) Drain and serve immediately."},
                {"role": "user", "content": "How much salt should I add to the water?"},
                {"role": "assistant", "content": "A good rule of thumb is to add about 1 tablespoon of salt per gallon of water, or roughly 1 teaspoon per quart. The water should taste like mild seawater. This seasons the pasta from the inside and makes a big difference in flavor!"},
                {"role": "user", "content": "What if I don't have enough salt?"},
                {"role": "assistant", "content": "If you're short on salt, use what you have - even a little bit helps. You can also compensate by being more generous with seasoning your sauce, or adding a pinch of salt directly to the pasta after cooking. The key is ensuring your final dish is well-seasoned overall."}
            ],
            "data_source": "cooking_dialogue",
            "ground_truth": "Practical cooking instructions and tips"
        },
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "ok"},
            ],
            "data_source": "low_quality_dialogue",
            "ground_truth": "Should be more engaging and helpful"
        }
    ]
    
    return sample_dialogues


def prepare_dialogue_dataset(
    input_file: str = None,
    output_dir: str = "./data",
    train_split: float = 0.8,
    create_sample: bool = False
) -> None:
    """
    Prepare dialogue dataset for training.
    
    Args:
        input_file: Path to input JSON file with dialogues
        output_dir: Directory to save output parquet files
        train_split: Fraction of data to use for training
        create_sample: Whether to create sample data instead of using input file
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if create_sample:
        print("Creating sample dialogue data...")
        dialogues = create_sample_dialogue_data()
    else:
        if not input_file:
            raise ValueError("Must provide input_file or set create_sample=True")
        
        print(f"Loading dialogues from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
    
    # Validate dialogue format
    print("Validating dialogue format...")
    valid_dialogues = []
    
    for i, dialogue in enumerate(dialogues):
        if not validate_dialogue_format(dialogue):
            print(f"Skipping invalid dialogue at index {i}")
            continue
        valid_dialogues.append(dialogue)
    
    print(f"Valid dialogues: {len(valid_dialogues)}")
    
    # Split into train and validation
    split_idx = int(len(valid_dialogues) * train_split)
    train_data = valid_dialogues[:split_idx]
    val_data = valid_dialogues[split_idx:]
    
    # Convert to DataFrames and save
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_path = output_path / "train.parquet"
    val_path = output_path / "test.parquet"
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    
    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")
    
    # Display sample
    print("\nSample training data:")
    print(train_df.head(2).to_string())


def validate_dialogue_format(dialogue: Dict[str, Any]) -> bool:
    """Validate that dialogue has the required format."""
    
    # Check required fields
    if "messages" not in dialogue:
        print("Missing 'messages' field")
        return False
    
    if not isinstance(dialogue["messages"], list):
        print("'messages' should be a list")
        return False
    
    if len(dialogue["messages"]) < 2:
        print("Dialogue should have at least 2 messages")
        return False
    
    # Check message format
    for i, message in enumerate(dialogue["messages"]):
        if not isinstance(message, dict):
            print(f"Message {i} should be a dict")
            return False
        
        if "role" not in message or "content" not in message:
            print(f"Message {i} missing 'role' or 'content'")
            return False
        
        if message["role"] not in ["user", "assistant", "system"]:
            print(f"Message {i} has invalid role: {message['role']}")
            return False
        
        if not isinstance(message["content"], str) or len(message["content"].strip()) == 0:
            print(f"Message {i} has empty or invalid content")
            return False
    
    # Set default values for optional fields
    if "data_source" not in dialogue:
        dialogue["data_source"] = "unknown"
    
    if "ground_truth" not in dialogue:
        dialogue["ground_truth"] = ""
    
    return True


def analyze_dataset(parquet_file: str) -> None:
    """Analyze the prepared dataset."""
    
    df = pd.read_parquet(parquet_file)
    
    print(f"\nDataset Analysis for {parquet_file}")
    print("=" * 50)
    print(f"Total dialogues: {len(df)}")
    
    # Analyze dialogue lengths
    dialogue_lengths = df['messages'].apply(len)
    print(f"Average dialogue length: {dialogue_lengths.mean():.1f} turns")
    print(f"Min dialogue length: {dialogue_lengths.min()} turns")
    print(f"Max dialogue length: {dialogue_lengths.max()} turns")
    
    # Analyze data sources
    if 'data_source' in df.columns:
        print(f"\nData sources:")
        print(df['data_source'].value_counts())
    
    # Show sample dialogue
    print(f"\nSample dialogue:")
    sample_dialogue = df.iloc[0]['messages']
    for i, message in enumerate(sample_dialogue[:4]):  # Show first 4 turns
        print(f"Turn {i+1} ({message['role']}): {message['content'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Prepare dialogue data for multi-turn training")
    parser.add_argument("--input", help="Input JSON file with dialogues")
    parser.add_argument("--output", default="./dialogue_data", help="Output directory")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--sample", action="store_true", help="Create sample data instead")
    parser.add_argument("--analyze", help="Analyze existing parquet file")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.analyze)
    else:
        prepare_dialogue_dataset(
            input_file=args.input,
            output_dir=args.output,
            train_split=args.train_split,
            create_sample=args.sample
        )
        
        # Analyze the created datasets
        train_file = Path(args.output) / "train.parquet"
        if train_file.exists():
            analyze_dataset(str(train_file))


if __name__ == "__main__":
    main()