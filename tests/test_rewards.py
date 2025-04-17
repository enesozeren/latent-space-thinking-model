import unittest
import sys
import os
from typing import List, Optional

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.rewards import accuracy_reward, format_reward


class TestRewardFunctions(unittest.TestCase):
    def test_accuracy_reward_correct(self):
        """Test accuracy reward with correct answers."""
        prompts = ["What is 2+2?", "What is 3+3?"]
        completions = ["<think>Adding 2 and 2</think><answer>\\boxed{4}</answer>", 
                      "<think>Adding 3 and 3</think><answer>\\boxed{6}</answer>"]
        answers = [4, 6]
        
        rewards = accuracy_reward(prompts=prompts, completions=completions, answer=answers)
        self.assertEqual(rewards, [1.0, 1.0])
    
    def test_accuracy_reward_incorrect(self):
        """Test accuracy reward with incorrect answers."""
        prompts = ["What is 2+2?", "What is 3+3?"]
        completions = ["<think>Adding 2 and 2</think><answer>\\boxed{5}</answer>", 
                      "<think>Adding 3 and 3</think><answer>\\boxed{7}</answer>"]
        answers = [4, 6]
        
        rewards = accuracy_reward(prompts=prompts, completions=completions, answer=answers)
        self.assertEqual(rewards, [0.0, 0.0])
    
    def test_accuracy_reward_mixed(self):
        """Test accuracy reward with mixed correct/incorrect answers."""
        prompts = ["What is 2+2?", "What is 3+3?"]
        completions = ["<think>Adding 2 and 2</think><answer>\\boxed{4}</answer>", 
                      "<think>Adding 3 and 3</think><answer>\\boxed{7}</answer>"]
        answers = [4, 6]
        
        rewards = accuracy_reward(prompts=prompts, completions=completions, answer=answers)
        self.assertEqual(rewards, [1.0, 0.0])
    
    def test_accuracy_reward_missing_answer(self):
        """Test accuracy reward when the answer tag is missing."""
        prompts = ["What is 2+2?"]
        completions = ["<think>Adding 2 and 2</think> The answer is 4"]
        answers = [4]
        
        rewards = accuracy_reward(prompts=prompts, completions=completions, answer=answers)
        self.assertEqual(rewards, [0.0])
    
    def test_accuracy_reward_whitespace(self):
        """Test accuracy reward with extra whitespace."""
        prompts = ["What is 2+2?"]
        completions = ["<think>Adding 2 and 2</think><answer> \\boxed{ 4 } </answer>"]
        answers = [4]
        
        rewards = accuracy_reward(prompts=prompts, completions=completions, answer=answers)
        self.assertEqual(rewards, [1.0])
    
    def test_accuracy_reward_dict_input(self):
        """Test accuracy reward with dictionary input."""
        prompts = ["What is 2+2?"]
        completions = [{"content": "<think>Adding 2 and 2</think><answer>\\boxed{4}</answer>"}]
        answers = [4]
        
        rewards = accuracy_reward(prompts=prompts, completions=completions, answer=answers)
        self.assertEqual(rewards, [1.0])
    
    def test_format_reward_correct(self):
        """Test format reward with correctly formatted completions."""
        completions = ["<think>Some thinking</think><answer>\\boxed{result}</answer>", 
                      "<think>More thinking\nMultiple lines</think><answer>\\boxed{42}</answer>"]
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [1.0, 1.0])
    
    def test_format_reward_incorrect(self):
        """Test format reward with incorrectly formatted completions."""
        completions = ["<think>Some thinking</think> <answer>result</answer>",  # Missing \\boxed{}
                      "The answer is \\boxed{42}",                             # Missing tags
                      "<answer>\\boxed{42}</answer><think>Thinking after answer</think>"]  # Wrong order
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [0.0, 0.0, 0.0])
    
    def test_format_reward_whitespace(self):
        """Test format reward with different whitespace patterns."""
        completions = [
            "<think>Some thinking</think>\n<answer>\\boxed{result}</answer>",
            "<think>More thinking</think>   <answer> \\boxed{42} </answer>"
        ]
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [1.0, 1.0])
    
    def test_format_reward_dict_input(self):
        """Test format reward with dictionary input."""
        completions = [{"content": "<think>Some thinking</think><answer>\\boxed{result}</answer>"}]
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [1.0])


if __name__ == "__main__":
    unittest.main()