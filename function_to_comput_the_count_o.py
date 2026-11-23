
def count_vowels(word: str) -> int:
    """
    Counts the number of vowels in a given word.

    This function iterates over each character in the word, checking if it is a vowel (both lowercase and uppercase). If it is a vowel, it increments a counter.

    Parameters:
    word (str): The string for which to count the vowels.

    Returns:
    int: The number of vowels in the given word.

    Examples:
    >>> count_vowels('vibyor')
    3
    >>> count_vowels('aeiouAEIOU')
    6
    >>> count_vowels('xyz')
    0

    Edge cases:
    - If the input is not a string, the function will throw an error.
    - If the input is an empty string, the function will return 0.
    """
    vowels = 'aeiou'
    count = 0
    for char in word:
        if char.lower() in vowels:
            count += 1
    return count

import unittest
from your_module import count_vowels  # Import the function to be tested

class TestCountVowels(unittest.TestCase):
    def test_basic_functionality(self):
        self.assertEqual(count_vowels('vibyor'), 3)
        self.assertEqual(count_vowels('aeiou'), 5)
        self.assertEqual(count_vowels('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5)

    def test_edge_cases(self):
        self.assertEqual(count_vowels(""), 0)
        self.assertEqual(count_vowels(None), 0)
        self.assertEqual(count_vowels('z'), 0)

    def test_error_cases(self):
        with self.assertRaises(TypeError):
            count_vowels(123)
        with self.assertRaises(TypeError):
            count_vowels([1, 2, 3])

    def test_various_input_scenarios(self):
        self.assertEqual(count_vowels("Hello, World!"), 3)
        self.assertEqual(count_vowels("Python is amazing"), 6)
        self.assertEqual(count_vowels("a"), 1)
        self.assertEqual(count_vowels("AAaaIIee"), 6)

if __name__ == '__main__':
    unittest.main()