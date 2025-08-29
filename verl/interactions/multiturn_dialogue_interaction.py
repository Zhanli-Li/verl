# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team  
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
import asyncio
from typing import Any, Optional, Dict, List
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MultiturnDialogueInteraction(BaseInteraction):
    """A multi-turn dialogue interaction that uses separate models for scoring and response generation.
    
    This interaction manages a conversation where:
    - A scoring model evaluates each turn of the dialogue 
    - A response model generates replies based on conversation context
    - The system tracks conversation state across multiple turns
    - Rewards are computed based on dialogue quality metrics
    
    Key features:
    - Per-turn scoring with a dedicated model
    - Context-aware response generation  
    - Configurable scoring criteria and thresholds
    - Support for conversation termination conditions
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        
        # Configuration for scoring model
        self.scoring_model_name = config.get("scoring_model", "default_scorer")
        self.scoring_prompt_template = config.get(
            "scoring_prompt_template", 
            "Rate this dialogue turn from 0.0 to 1.0 based on relevance, helpfulness, and coherence:\n\nDialogue:\n{dialogue}\n\nScore:"
        )
        
        # Configuration for response model  
        self.response_model_name = config.get("response_model", "default_responder")
        self.response_prompt_template = config.get(
            "response_prompt_template",
            "Continue this conversation naturally and helpfully:\n\n{dialogue}\n\nResponse:"
        )
        
        # Scoring thresholds
        self.min_score_threshold = config.get("min_score_threshold", 0.3)
        self.good_score_threshold = config.get("good_score_threshold", 0.7)
        self.max_turns = config.get("max_turns", 10)
        
        # Weight for different turn positions (early turns might be weighted differently)
        self.turn_weights = config.get("turn_weights", [1.0] * self.max_turns)
        
        logger.info(f"Initialized MultiturnDialogueInteraction with scoring_model={self.scoring_model_name}, "
                   f"response_model={self.response_model_name}, max_turns={self.max_turns}")

    async def start_interaction(
        self, instance_id: Optional[str] = None, initial_context: Optional[str] = None, **kwargs
    ) -> str:
        """Start a new dialogue interaction instance.
        
        Args:
            instance_id: Unique identifier for this interaction instance
            initial_context: Optional context to start the dialogue
            **kwargs: Additional parameters
            
        Returns:
            The instance ID for this interaction
        """
        if instance_id is None:
            instance_id = str(uuid4())
            
        self._instance_dict[instance_id] = {
            "dialogue_history": [],  # List of {"role": "user/assistant", "content": "...", "score": float}
            "current_turn": 0,
            "total_score": 0.0,
            "average_score": 0.0,
            "initial_context": initial_context or "",
            "last_response": "",
            "conversation_active": True
        }
        
        logger.debug(f"Started interaction {instance_id} with context: {initial_context}")
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Generate a response for the current turn and score it.
        
        Args:
            instance_id: The interaction instance ID
            messages: Current conversation messages
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (should_terminate, response_content, turn_score, metadata)
        """
        if instance_id not in self._instance_dict:
            logger.error(f"Instance {instance_id} not found")
            return True, "Error: Instance not found", 0.0, {"error": "instance_not_found"}
        
        instance = self._instance_dict[instance_id]
        
        # Extract the latest assistant response from messages
        latest_response = ""
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            if message.get("role") == "assistant":
                latest_response = message.get("content", "")
                break
        
        instance["last_response"] = latest_response
        instance["current_turn"] += 1
        
        # Add current response to dialogue history
        if latest_response:
            instance["dialogue_history"].append({
                "role": "assistant", 
                "content": latest_response,
                "turn": instance["current_turn"]
            })
        
        # Score the current turn
        turn_score = await self._score_dialogue_turn(instance_id, messages)
        
        # Update instance scoring
        instance["total_score"] += turn_score
        instance["average_score"] = instance["total_score"] / instance["current_turn"]
        
        # Add score to the last dialogue entry
        if instance["dialogue_history"]:
            instance["dialogue_history"][-1]["score"] = turn_score
        
        # Determine if conversation should continue
        should_terminate = await self._should_terminate_conversation(instance_id)
        
        if should_terminate:
            response_content = await self._generate_termination_response(instance_id)
            instance["conversation_active"] = False
        else:
            response_content = await self._generate_continuation_response(instance_id, messages)
        
        # Metadata for tracking
        metadata = {
            "turn": instance["current_turn"],
            "turn_score": turn_score,
            "average_score": instance["average_score"],
            "total_score": instance["total_score"], 
            "should_terminate": should_terminate,
            "dialogue_length": len(instance["dialogue_history"])
        }
        
        logger.debug(f"Turn {instance['current_turn']} for {instance_id}: score={turn_score:.3f}, "
                    f"avg={instance['average_score']:.3f}, terminate={should_terminate}")
        
        return should_terminate, response_content, turn_score, metadata

    async def _score_dialogue_turn(self, instance_id: str, messages: list[dict[str, Any]]) -> float:
        """Score the current dialogue turn using the scoring model.
        
        This is a placeholder implementation. In a real scenario, you would:
        1. Format the dialogue context for the scoring model
        2. Call the scoring model API 
        3. Parse and normalize the score
        
        Args:
            instance_id: The interaction instance ID
            messages: Current conversation messages
            
        Returns:
            Score between 0.0 and 1.0
        """
        instance = self._instance_dict[instance_id]
        
        # Format dialogue context for scoring
        dialogue_text = self._format_dialogue_for_scoring(messages)
        
        # Placeholder scoring logic - in practice, call your scoring model here
        # For now, we simulate scoring based on response length and content quality
        if not instance["last_response"]:
            return 0.0
        
        response = instance["last_response"]
        base_score = 0.5
        
        # Length bonus (reasonable length responses score higher)
        length_score = min(len(response) / 200.0, 0.3)  # Up to +0.3 for ~200 chars
        
        # Content quality heuristics (replace with actual model scoring)
        quality_score = 0.0
        if any(word in response.lower() for word in ["help", "understand", "explain", "because"]):
            quality_score += 0.1
        if "?" in response:  # Asking clarifying questions
            quality_score += 0.1
        if len(response.split()) > 5:  # Substantial response
            quality_score += 0.1
            
        final_score = min(base_score + length_score + quality_score, 1.0)
        
        # Add some randomness to simulate model variation
        import random
        random.seed(hash(response + str(instance["current_turn"])))  # Deterministic for testing
        final_score += random.uniform(-0.1, 0.1)
        final_score = max(0.0, min(1.0, final_score))
        
        logger.debug(f"Scored turn {instance['current_turn']}: {final_score:.3f} "
                    f"(base={base_score}, length={length_score:.3f}, quality={quality_score:.3f})")
        
        return final_score

    async def _should_terminate_conversation(self, instance_id: str) -> bool:
        """Determine if the conversation should be terminated.
        
        Args:
            instance_id: The interaction instance ID
            
        Returns:
            True if conversation should end
        """
        instance = self._instance_dict[instance_id]
        
        # Terminate if max turns reached
        if instance["current_turn"] >= self.max_turns:
            logger.debug(f"Terminating {instance_id}: max turns ({self.max_turns}) reached")
            return True
        
        # Terminate if recent scores are consistently low
        if instance["current_turn"] >= 3:
            recent_scores = [entry.get("score", 0.0) for entry in instance["dialogue_history"][-3:]]
            if all(score < self.min_score_threshold for score in recent_scores):
                logger.debug(f"Terminating {instance_id}: consistently low scores {recent_scores}")
                return True
        
        # Terminate if average score is very high (successful completion)
        if instance["current_turn"] >= 2 and instance["average_score"] > self.good_score_threshold:
            logger.debug(f"Terminating {instance_id}: high average score {instance['average_score']:.3f}")
            return True
        
        return False

    async def _generate_continuation_response(self, instance_id: str, messages: list[dict[str, Any]]) -> str:
        """Generate a response to continue the conversation.
        
        This is a placeholder implementation. In practice, you would:
        1. Format the conversation context for the response model
        2. Call the response model API
        3. Return the generated response
        
        Args:
            instance_id: The interaction instance ID
            messages: Current conversation messages
            
        Returns:
            Generated response content
        """
        instance = self._instance_dict[instance_id]
        
        # Analyze the dialogue quality so far
        if instance["average_score"] < self.min_score_threshold:
            return ("I notice our conversation might be going off track. "
                   "Let me try to provide a more helpful response. Can you clarify what you're looking for?")
        elif instance["average_score"] > self.good_score_threshold:
            return ("Great! This conversation is going well. What else would you like to discuss or explore?")
        else:
            return ("I understand. Let me think about this more carefully and provide a better response.")

    async def _generate_termination_response(self, instance_id: str) -> str:
        """Generate a response when terminating the conversation.
        
        Args:
            instance_id: The interaction instance ID
            
        Returns:
            Termination response content
        """
        instance = self._instance_dict[instance_id]
        
        if instance["average_score"] > self.good_score_threshold:
            return "This has been a productive conversation! Thank you for the engaging dialogue."
        elif instance["current_turn"] >= self.max_turns:
            return "We've reached the end of our conversation time. Thank you for chatting!"
        else:
            return "Let's wrap up this conversation here. Thank you for your time."

    def _format_dialogue_for_scoring(self, messages: list[dict[str, Any]]) -> str:
        """Format the dialogue history for the scoring model.
        
        Args:
            messages: Current conversation messages
            
        Returns:
            Formatted dialogue text
        """
        formatted_lines = []
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            formatted_lines.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted_lines)

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate the overall score for the dialogue interaction.
        
        Args:
            instance_id: The interaction instance ID
            **kwargs: Additional parameters
            
        Returns:
            Overall dialogue score
        """
        if instance_id not in self._instance_dict:
            return 0.0
        
        instance = self._instance_dict[instance_id]
        return instance["average_score"]

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up the interaction instance.
        
        Args:
            instance_id: The interaction instance ID
            **kwargs: Additional parameters
        """
        if instance_id in self._instance_dict:
            final_score = self._instance_dict[instance_id]["average_score"]
            logger.info(f"Finalized interaction {instance_id} with final score: {final_score:.3f}")
            del self._instance_dict[instance_id]
        else:
            logger.warning(f"Attempted to finalize non-existent instance: {instance_id}")