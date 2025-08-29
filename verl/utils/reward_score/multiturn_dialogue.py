# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom reward functions for multi-turn dialogue training.

This module provides reward functions that can be used with verl's PPO training
for multi-turn conversational AI. The functions integrate with external scoring
models to evaluate dialogue quality.
"""

import re
import json
import logging
from typing import Optional, Dict, Any, List
import asyncio
import httpx
import os

logger = logging.getLogger(__name__)


def compute_multiturn_dialogue_score(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute reward score for multi-turn dialogue using a scoring model.
    
    This function evaluates the quality of a multi-turn dialogue response
    by analyzing various aspects like coherence, relevance, helpfulness,
    and conversational flow.
    
    Args:
        data_source: The dataset/source identifier
        solution_str: The model's generated response/dialogue
        ground_truth: Expected or reference response (may be None for open dialogue)
        extra_info: Additional context information
        
    Returns:
        Float score between 0.0 and 1.0
    """
    if not solution_str or solution_str.strip() == "":
        return 0.0
    
    # Parse dialogue turns if the solution contains multi-turn structure
    dialogue_turns = _parse_dialogue_turns(solution_str)
    
    # Base scoring components
    coherence_score = _evaluate_coherence(dialogue_turns)
    relevance_score = _evaluate_relevance(dialogue_turns, ground_truth)
    helpfulness_score = _evaluate_helpfulness(dialogue_turns)
    conversation_flow_score = _evaluate_conversation_flow(dialogue_turns)
    
    # Weighted combination of scores
    weights = {
        'coherence': 0.3,
        'relevance': 0.3, 
        'helpfulness': 0.25,
        'conversation_flow': 0.15
    }
    
    final_score = (
        weights['coherence'] * coherence_score +
        weights['relevance'] * relevance_score +
        weights['helpfulness'] * helpfulness_score +
        weights['conversation_flow'] * conversation_flow_score
    )
    
    # Apply bonuses/penalties
    if len(dialogue_turns) > 1:  # Multi-turn bonus
        final_score *= 1.1
    
    if _contains_inappropriate_content(solution_str):  # Safety penalty
        final_score *= 0.5
    
    # Normalize to [0, 1]
    final_score = max(0.0, min(1.0, final_score))
    
    logger.debug(f"Dialogue score: {final_score:.3f} "
                f"(coherence={coherence_score:.3f}, relevance={relevance_score:.3f}, "
                f"helpfulness={helpfulness_score:.3f}, flow={conversation_flow_score:.3f})")
    
    return final_score


async def compute_model_based_dialogue_score(
    data_source: str,
    solution_str: str, 
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    scoring_model_url: str = "http://localhost:8000/v1/chat/completions",
    scoring_model_name: str = "gpt-3.5-turbo"
) -> float:
    """
    Compute dialogue score using an external scoring model API.
    
    This function calls an external model (like GPT, Claude, or a custom model)
    to score the dialogue quality. This is more accurate than rule-based scoring
    but requires an API endpoint.
    
    Args:
        data_source: The dataset/source identifier
        solution_str: The model's generated response/dialogue
        ground_truth: Expected or reference response
        extra_info: Additional context information
        scoring_model_url: URL of the scoring model API
        scoring_model_name: Name/identifier of the scoring model
        
    Returns:
        Float score between 0.0 and 1.0
    """
    if not solution_str or solution_str.strip() == "":
        return 0.0
    
    try:
        # Prepare the scoring prompt
        scoring_prompt = _create_scoring_prompt(solution_str, ground_truth, data_source)
        
        # Call the scoring model
        score = await _call_scoring_model(
            scoring_prompt, 
            scoring_model_url, 
            scoring_model_name
        )
        
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        logger.warning(f"Failed to get model-based score, falling back to rule-based: {e}")
        # Fallback to rule-based scoring
        return compute_multiturn_dialogue_score(data_source, solution_str, ground_truth, extra_info)


def _parse_dialogue_turns(solution_str: str) -> List[Dict[str, str]]:
    """Parse dialogue turns from the solution string."""
    turns = []
    
    # Try to parse structured dialogue format first
    # Format: "User: ... Assistant: ... User: ... Assistant: ..."
    pattern = r'(User|Assistant|Human|AI):\s*([^U^A^H]*?)(?=(?:User|Assistant|Human|AI):|$)'
    matches = re.findall(pattern, solution_str, re.IGNORECASE | re.DOTALL)
    
    if matches:
        for role, content in matches:
            content = content.strip()
            if content:
                turns.append({
                    'role': role.lower(),
                    'content': content
                })
    else:
        # If no structured format, treat entire text as single response
        turns.append({
            'role': 'assistant',
            'content': solution_str.strip()
        })
    
    return turns


def _evaluate_coherence(dialogue_turns: List[Dict[str, str]]) -> float:
    """Evaluate the coherence of the dialogue."""
    if not dialogue_turns:
        return 0.0
    
    coherence_score = 0.3  # Base score
    
    for turn in dialogue_turns:
        content = turn['content']
        
        # Check for basic coherence indicators
        if len(content.split()) >= 5:  # Meaningful length
            coherence_score += 0.3
        elif len(content.split()) >= 3:
            coherence_score += 0.2
        
        # Check for complete sentences
        if content.endswith('.') or content.endswith('!') or content.endswith('?'):
            coherence_score += 0.2
        
        # Check for logical connectors
        connectors = ['because', 'therefore', 'however', 'moreover', 'also', 'furthermore']
        if any(conn in content.lower() for conn in connectors):
            coherence_score += 0.2
        
        # Check for proper grammar indicators
        if any(word in content.lower() for word in ['the', 'and', 'to', 'of', 'a']):
            coherence_score += 0.1
    
    return min(1.0, coherence_score / len(dialogue_turns))


def _evaluate_relevance(dialogue_turns: List[Dict[str, str]], ground_truth: str) -> float:
    """Evaluate relevance to the context/ground truth."""
    if not dialogue_turns:
        return 0.0
    
    if not ground_truth:
        # If no ground truth, use heuristics
        return _evaluate_relevance_heuristic(dialogue_turns)
    
    relevance_score = 0.0
    
    # Simple keyword overlap with ground truth
    ground_truth_words = set(ground_truth.lower().split())
    
    for turn in dialogue_turns:
        content_words = set(turn['content'].lower().split())
        overlap = len(ground_truth_words.intersection(content_words))
        relevance_score += min(0.5, overlap / max(len(ground_truth_words), 1))
    
    return min(1.0, relevance_score / len(dialogue_turns))


def _evaluate_relevance_heuristic(dialogue_turns: List[Dict[str, str]]) -> float:
    """Evaluate relevance using heuristics when no ground truth is available."""
    base_score = 0.5
    
    # Check for question-answering patterns
    for i, turn in enumerate(dialogue_turns):
        content = turn['content'].lower()
        
        # Responding to questions
        if i > 0 and '?' in dialogue_turns[i-1]['content']:
            if any(word in content for word in ['yes', 'no', 'because', 'the answer is']):
                base_score += 0.2
        
        # Asking clarifying questions
        if '?' in content and any(word in content for word in ['what', 'how', 'why', 'when', 'where']):
            base_score += 0.1
    
    return min(1.0, base_score)


def _evaluate_helpfulness(dialogue_turns: List[Dict[str, str]]) -> float:
    """Evaluate how helpful the dialogue is."""
    if not dialogue_turns:
        return 0.0
    
    helpfulness_score = 0.2  # Base score
    
    for turn in dialogue_turns:
        content = turn['content'].lower()
        
        # Helpful phrases
        helpful_phrases = [
            'let me help', 'i can help', 'here\'s how', 'the solution is',
            'you can', 'try this', 'here\'s what', 'i recommend', 'i suggest',
            'of course', 'sure', 'absolutely', 'i\'d be happy to'
        ]
        if any(phrase in content for phrase in helpful_phrases):
            helpfulness_score += 0.3
        
        # Providing explanations
        explanation_words = ['because', 'since', 'therefore', 'this means', 'in other words', 'explanation']
        if any(word in content for word in explanation_words):
            helpfulness_score += 0.2
        
        # Offering alternatives or next steps
        alternative_phrases = ['alternatively', 'another option', 'you could also', 'next steps']
        if any(phrase in content for phrase in alternative_phrases):
            helpfulness_score += 0.2
        
        # Educational content
        educational_words = ['learn', 'understand', 'explain', 'example', 'tutorial', 'guide']
        if any(word in content for word in educational_words):
            helpfulness_score += 0.2
    
    return min(1.0, helpfulness_score / len(dialogue_turns))


def _evaluate_conversation_flow(dialogue_turns: List[Dict[str, str]]) -> float:
    """Evaluate the natural flow of conversation."""
    if len(dialogue_turns) <= 1:
        return 1.0  # Single turn is automatically "flowing"
    
    flow_score = 0.5  # Base score
    
    for i in range(1, len(dialogue_turns)):
        current_turn = dialogue_turns[i]['content'].lower()
        previous_turn = dialogue_turns[i-1]['content'].lower()
        
        # Check for natural transitions
        transition_phrases = [
            'speaking of', 'that reminds me', 'on that note', 'following up',
            'to add to that', 'building on', 'regarding', 'about'
        ]
        if any(phrase in current_turn for phrase in transition_phrases):
            flow_score += 0.2
        
        # Check for acknowledgment of previous turn
        acknowledgments = ['i see', 'i understand', 'that makes sense', 'interesting', 'right']
        if any(ack in current_turn for ack in acknowledgments):
            flow_score += 0.1
        
        # Penalize abrupt topic changes (very simplistic check)
        current_words = set(current_turn.split())
        previous_words = set(previous_turn.split())
        overlap = len(current_words.intersection(previous_words))
        if overlap < 2 and len(current_words) > 5 and len(previous_words) > 5:
            flow_score -= 0.1
    
    return max(0.0, min(1.0, flow_score))


def _contains_inappropriate_content(text: str) -> bool:
    """Check for inappropriate content that should be penalized."""
    inappropriate_patterns = [
        r'\b(hate|violence|harmful)\b',
        r'(kill|destroy|hurt)',
        # Add more patterns as needed
    ]
    
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in inappropriate_patterns)


def _create_scoring_prompt(solution_str: str, ground_truth: str, data_source: str) -> str:
    """Create a prompt for the scoring model."""
    prompt = f"""Please evaluate the following dialogue response on a scale from 0.0 to 1.0 based on:
1. Coherence and clarity
2. Relevance to the conversation
3. Helpfulness to the user
4. Natural conversation flow

Dialogue Response:
{solution_str}

{f"Reference/Expected Response: {ground_truth}" if ground_truth else ""}

Data Source: {data_source}

Please provide only a numerical score between 0.0 and 1.0."""
    
    return prompt


async def _call_scoring_model(prompt: str, api_url: str, model_name: str) -> float:
    """Call the external scoring model API."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Extract numerical score from response
                score_match = re.search(r'(\d+\.?\d*)', content)
                if score_match:
                    score = float(score_match.group(1))
                    # Normalize if score is out of range
                    if score > 1.0:
                        score = score / 10.0 if score <= 10.0 else 1.0
                    return score
                else:
                    logger.warning(f"Could not parse score from model response: {content}")
                    return 0.5
            else:
                logger.warning(f"Scoring model API error: {response.status_code}")
                return 0.5
                
    except Exception as e:
        logger.error(f"Error calling scoring model: {e}")
        return 0.5


# Alias for backward compatibility and easier configuration
compute_score = compute_multiturn_dialogue_score