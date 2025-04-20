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
        completions = [
            "<think>\nSolving 2+2\n</think>\n<answer>\n$\\boxed{4}$\n</answer>",
            "<think>\nSolving 3*2\n</think>\n<answer>\n$\\boxed{6}$\n</answer>"
        ]
        answers = ["4", "6"]
        
        rewards = accuracy_reward(completions=completions, answers=answers)
        self.assertEqual(rewards, [1.0, 1.0])
    
    def test_accuracy_reward_incorrect(self):
        """Test accuracy reward with incorrect answers."""
        completions = [
            "<think>\nSolving 2+2\n</think>\n<answer>\n$\\boxed{5}$\n</answer>",
            "<think>\nSolving 3*2\n</think>\n<answer>\n$\\boxed{7}$\n</answer>"
        ]
        answers = ["4", "6"]
        
        rewards = accuracy_reward(completions=completions, answers=answers)
        self.assertEqual(rewards, [0.0, 0.0])
    
    def test_accuracy_reward_unparseable(self):
        """Test accuracy reward with unparseable answers."""
        completions = ["<think>\nSolving 2+2\n</think>\n<answer>\nThe answer is 4\n</answer>"]
        answers = ["4"]
        
        rewards = accuracy_reward(completions=completions, answers=answers)
        # The answer "The answer is 4" can be parsed but doesn't match the gold answer,
        # so we expect 0.0 instead of None
        self.assertEqual(rewards, [0.0])
    
    def test_accuracy_reward_unparseable_solution(self):
        """Test accuracy reward with unparseable solution returns None."""
        completions = ["<think>\nSolving\n</think>\n<answer>\n$\\boxed{4}$\n</answer>"]
        answers = ["unparseable text"]
        
        rewards = accuracy_reward(completions=completions, answers=answers)
        self.assertIsNone(rewards[0])
    
    def test_accuracy_reward_mathematically_equivalent(self):
        """Test accuracy reward with mathematically equivalent answers."""
        completions = [
            # Testing equivalent fractions
            "<think>\nSimplifying 4/8\n</think>\n<answer>\n$\\boxed{1/2}$\n</answer>",
            # Testing equivalent expressions
            "<think>\nCalculating 2(3+4)\n</think>\n<answer>\n$\\boxed{14}$\n</answer>",
            # Testing different forms of the same number
            "<think>\nComputing 3^2\n</think>\n<answer>\n$\\boxed{9}$\n</answer>",
            # Testing alternative representation with decimal
            "<think>\nCalculating 5/4\n</think>\n<answer>\n$\\boxed{1.25}$\n</answer>"
        ]
        
        answers = ["4/8", "2*7", "3^2", "5/4"]
        
        rewards = accuracy_reward(completions=completions, answers=answers)
        self.assertEqual(rewards, [1.0, 1.0, 1.0, 1.0])
    
    def test_accuracy_reward_complex_equivalence(self):
        """Test accuracy reward with more complex equivalent expressions."""
        completions = [
            # Testing trigonometric identity
            "<think>\nUsing trigonometric identity\n</think>\n<answer>\n$\\boxed{\\sin^2(x) + \\cos^2(x)}$\n</answer>",
            # Testing alternative forms of expressions
            "<think>\nComputing different form\n</think>\n<answer>\n$\\boxed{2+3}$\n</answer>",
            # Testing different forms of the same number
            "<think>\nComputing in a different way\n</think>\n<answer>\n$\\boxed{2*3}$\n</answer>"
        ]
        
        answers = ["1", "5", "6"]
        
        rewards = accuracy_reward(completions=completions, answers=answers)
        self.assertEqual(rewards, [1.0, 1.0, 1.0])
        
    def test_format_reward_correct(self):
        """Test format reward with correctly formatted completions."""
        completions = [
            "<think>\nSome thinking\n</think>\n<answer>\n$\\boxed{result}$\n</answer>",
            "<think>\nMore thinking\nMultiple lines\n</think>\n<answer>\n$\\boxed{42}$\n</answer>"
        ]
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [1.0, 1.0])
    
    def test_format_reward_partial(self):
        """Test format reward with partially formatted completions (no boxed)."""
        completions = [
            "<think>\nSome thinking\n</think>\n<answer>\nresult without boxed\n</answer>",
            "<think>\nMore thinking\n</think>\n<answer>\n42\n</answer>"
        ]
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [0.5, 0.5])
    
    def test_format_reward_incorrect(self):
        """Test format reward with incorrectly formatted completions."""
        completions = [
            "Thinking first\n<answer>\n$\\boxed{result}$\n</answer>",  # Missing think tags
            "<think>\nSome thinking\n</think>\nThe answer is 42",  # Missing answer tags
            "<answer>\n$\\boxed{42}$\n</answer>\n<think>\nThinking after answer\n</think>"  # Wrong order
        ]
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [0.0, 0.0, 0.0])
    
    def test_format_reward_pattern_matching(self):
        """Test format reward pattern matching with whitespace restrictions."""
        completions = [
            "<think>\nMulti-line\nthinking\n</think>\n<answer>\n$\\boxed{result}$\n</answer>",  # Should pass (newline is allowed whitespace)
            "<think>No newline</think> <answer>\n$\\boxed{42}$\n</answer>",  # Should pass (space is allowed whitespace)
            "<think>\nThinking\n</think><answer>\n$\\boxed{42}$\n</answer>",  # Should pass (no whitespace between tags is still valid)
            "<think>\nThinking\n</think>\t<answer>\n$\\boxed{42}$\n</answer>"  # Should pass (tab is allowed whitespace)
        ]
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [1.0, 1.0, 1.0, 1.0])
    
    def test_format_reward_invalid_content_between_tags(self):
        """Test format reward when there is non-whitespace content between </think> and <answer> tags."""
        completions = [
            "<think>\nThinking\n</think>some text<answer>\n$\\boxed{42}$\n</answer>",  # Should fail (non-whitespace between tags)
            "<think>\nThinking\n</think>123<answer>\n$\\boxed{42}$\n</answer>",  # Should fail (numbers between tags)
            "<think>\nThinking\n</think><!-- comment --><answer>\n$\\boxed{42}$\n</answer>"  # Should fail (comment between tags)
        ]
        
        rewards = format_reward(completions=completions)
        self.assertEqual(rewards, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()