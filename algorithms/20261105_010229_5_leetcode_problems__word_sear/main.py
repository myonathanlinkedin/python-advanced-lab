import unittest
from core import WordSearchSolver, Trie, count_palindromic_substrings

class TestWordSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = WordSearchSolver()

    def test_basic_true(self):
        board = [
            ['A','B','C','E'],
            ['S','F','C','S'],
            ['A','D','E','E']
        ]
        self.assertTrue(self.solver.exist(board, "ABCCED"))
        self.assertTrue(self.solver.exist(board, "SEE"))
        self.assertTrue(self.solver.exist(board, "ABCB"))

    def test_basic_false(self):
        board = [
            ['A','B'],
            ['C','D']
        ]
        self.assertFalse(self.solver.exist(board, "ACDB"))
        self.assertFalse(self.solver.exist(board, "ABCD"))

    def test_single_cell(self):
        board = [['a']]
        self.assertTrue(self.solver.exist(board, "a"))
        self.assertFalse(self.solver.exist(board, "b"))

    def test_empty_inputs(self):
        self.assertFalse(self.solver.exist([], "a"))
        self.assertFalse(self.solver.exist([[]], "a"))
        self.assertFalse(self.solver.exist([['a']], ""))

class TestTrie(unittest.TestCase):
    def setUp(self) -> None:
        self.trie = Trie()
        for word in ["apple", "app", "application", "bat", "batch"]:
            self.trie.insert(word)

    def test_search(self):
        self.assertTrue(self.trie.search("apple"))
        self.assertTrue(self.trie.search("app"))
        self.assertFalse(self.trie.search("appl"))
        self.assertFalse(self.trie.search("batt"))

    def test_startsWith(self):
        self.assertTrue(self.trie.startsWith("app"))
        self.assertTrue(self.trie.startsWith("bat"))
        self.assertFalse(self.trie.startsWith("cat"))
        self.assertTrue(self.trie.startsWith(""))  # empty prefix always true

    def test_insert_and_search_empty_string(self):
        empty_trie = Trie()
        empty_trie.insert("")
        self.assertTrue(empty_trie.search(""))
        self.assertTrue(empty_trie.startsWith(""))

class TestPalindromicSubstrings(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(count_palindromic_substrings("abc"), 3)   # a,b,c
        self.assertEqual(count_palindromic_substrings("aaa"), 6)   # a,a,a,aa,aa,aaa
        self.assertEqual(count_palindromic_substrings(""), 0)
        self.assertEqual(count_palindromic_substrings("a"), 1)

    def test_mixed(self):
        self.assertEqual(count_palindromic_substrings("ababa"), 9)
        # a,b,a,b,a,aba,bab,aba,ababa

    def test_long_random(self):
        s = "abcd" * 25  # length 100, no multi‑char palindromes
        self.assertEqual(count_palindromic_substrings(s), 100)  # each single char

if __name__ == '__main__':
    unittest.main()
