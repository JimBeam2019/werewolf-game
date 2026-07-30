import unittest

from infrastructure.langchain_speak_strategy import _shorten


class TestShorten(unittest.TestCase):
    def test_short_message_passes_through_unchanged(self):
        self.assertEqual(_shorten("yeah agreed."), "yeah agreed.")

    def test_two_short_sentences_pass_through_unchanged(self):
        text = "I think Bob is suspicious. He's been dodging questions."
        self.assertEqual(_shorten(text), text)

    def test_long_single_sentence_gets_truncated(self):
        text = (
            "Based on my observations, it appears that Carol has been "
            "unusually quiet throughout this entire conversation, which "
            "I find rather suspicious given the circumstances at hand."
        )
        result = _shorten(text)
        self.assertLess(len(result.split()), len(text.split()))
        self.assertTrue(result.endswith("..."))

    def test_more_than_two_sentences_gets_capped_at_two(self):
        text = "First sentence here. Second sentence here. Third sentence should be dropped."
        result = _shorten(text)
        self.assertNotIn("Third sentence", result)

    def test_strips_surrounding_quotes(self):
        self.assertEqual(_shorten('"nah I dont buy it."'), "nah I dont buy it.")

    def test_empty_string_does_not_raise(self):
        self.assertEqual(_shorten(""), "")

    def test_strips_a_formal_opener(self):
        result = _shorten("In my opinion, Frank is acting weird.")
        self.assertEqual(result, "Frank is acting weird.")

    def test_strips_stacked_formal_openers(self):
        # Models often chain more than one formal opener in a row -
        # stripping only the first would still leave the discussion
        # sounding stiff.
        text = "Based on my observations, it appears that Carol is lying."
        result = _shorten(text)
        self.assertEqual(result, "Carol is lying.")

    def test_casual_text_without_an_opener_is_left_untouched(self):
        # Regression test: the re-capitalization step that fixes up text
        # after stripping an opener must not fire when nothing was
        # stripped - otherwise intentionally-casual lowercase openers
        # like "nah..." get needlessly capitalized into something stiffer.
        self.assertEqual(_shorten("nah I dont buy it"), "nah I dont buy it")

    def test_word_cap_matches_the_15_word_limit_stated_in_the_prompt(self):
        text = " ".join(f"word{i}" for i in range(30)) + "."
        result = _shorten(text)
        word_count = len(result.rstrip(".").split())
        self.assertLessEqual(word_count, 15)


if __name__ == "__main__":
    unittest.main()
