import unittest
from typing import List, Optional
from unittest.mock import patch

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from train.rewards import accuracy_reward, format_reward


class TestRewardFunctions(unittest.TestCase):
    def setUp(self):
        # dummy prompts (unused by accuracy_reward)
        self.prompts = ["p1", "p2"]

    def test_accuracy_reward_no_answer_tag(self):
        completions = ["no tags here"]
        answers = ["42"]
        rewards = accuracy_reward(prompts=self.prompts, completions=completions, answer=answers)
        self.assertEqual(rewards, [0.0])

    def test_accuracy_reward_multiple_answer_tags(self):
        # now that we zero out on !=1 <answer> block
        comp = "<answer>1</answer> some text <answer>2</answer>"
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["1"])
        self.assertEqual(rewards, [0.0])

    def test_accuracy_reward_no_boxed(self):
        comp = "<answer>some content without box</answer>"
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["123"])
        self.assertEqual(rewards, [0.0])

    def test_accuracy_reward_numeric_equal(self):
        comp = '<answer> \\boxed{  42 } </answer>'
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["42"])
        self.assertEqual(rewards, [1.0])

    @patch("train.rewards.parse", autospec=True)
    @patch("train.rewards.verify", autospec=True)
    def test_accuracy_reward_symbolic_equal(self, mock_verify, mock_parse):
        sol_obj = object()
        content_obj = object()
        mock_parse.side_effect = [sol_obj, content_obj]
        mock_verify.return_value = True

        comp = '<answer> text \\boxed{84/2} end</answer>'
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["42"])
        self.assertEqual(rewards, [0.5])
        self.assertEqual(mock_parse.call_count, 2)
        mock_verify.assert_called_once_with(sol_obj, content_obj)

    @patch("train.rewards.parse", autospec=True)
    @patch("train.rewards.verify", autospec=True)
    def test_accuracy_reward_symbolic_not_equal(self, mock_verify, mock_parse):
        sol_obj = object()
        content_obj = object()
        mock_parse.side_effect = [sol_obj, content_obj]
        mock_verify.return_value = False

        comp = '<answer> \\boxed{x+1} </answer>'
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["x+2"])
        self.assertEqual(rewards, [0.0])

    @patch("train.rewards.parse", autospec=True)
    def test_accuracy_reward_parse_fail_on_solution(self, mock_parse):
        mock_parse.return_value = None
        comp = '<answer> \\boxed{foo} </answer>'
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["bar"])
        self.assertEqual(rewards, [None])

    @patch("train.rewards.parse", autospec=True)
    def test_accuracy_reward_parse_fail_on_content(self, mock_parse):
        # first parse(sol) ok, second parse(content) returns None → reward 0.0 now
        dummy = object()
        mock_parse.side_effect = [dummy, None]
        comp = '<answer> \\boxed{foo} </answer>'
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["foo"])
        self.assertEqual(rewards, [0.0])

    @patch("train.rewards.parse", autospec=True)
    @patch("train.rewards.verify", autospec=True)
    def test_accuracy_reward_verify_exception(self, mock_verify, mock_parse):
        sol_obj = object()
        content_obj = object()
        mock_parse.side_effect = [sol_obj, content_obj]
        mock_verify.side_effect = RuntimeError("boom")

        comp = '<answer> \\boxed{anything} </answer>'
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["anything"])
        self.assertEqual(rewards, [0.0])

    def test_accuracy_reward_dict_based(self):
        # ensure dict completions work the same
        comp = {"content": "<answer> \\boxed{7} </answer>"}
        rewards = accuracy_reward(prompts=self.prompts, completions=[comp], answer=["7"])
        self.assertEqual(rewards, [1.0])

    def test_format_reward_full_match(self):
        comp = "<think> reasoning </think>\n<answer> result is \\boxed{42} </answer>"
        rewards = format_reward([comp])
        self.assertEqual(rewards, [1.0])

    def test_format_reward_partial_match(self):
        comp = "<think> hi </think>   <answer> just an answer </answer>"
        rewards = format_reward([comp])
        self.assertEqual(rewards, [0.5])

    def test_format_reward_no_think(self):
        comp = "<answer> \\boxed{42} </answer>"
        rewards = format_reward([comp])
        self.assertEqual(rewards, [0.0])

    def test_format_reward_extra_content_between_tags(self):
        comp = "<think> a </think> EXTRA <answer> \\boxed{1} </answer>"
        rewards = format_reward([comp])
        self.assertEqual(rewards, [0.0])

    def test_format_reward_multiple_completions(self):
        comps = [
            "<think>x</think><answer>\\boxed{1}</answer>",
            "<think>y</think><answer>no box</answer>",
            "just text"
        ]
        rewards = format_reward(comps)
        self.assertEqual(rewards, [1.0, 0.5, 0.0])


if __name__ == "__main__":
    unittest.main()