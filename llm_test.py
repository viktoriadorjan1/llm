import unittest
from llm import chunk_text, re_rank_chunks


class TestTextProcessing(unittest.TestCase):

    def test_chunk_text(self):
        # Test case 1: A general chunking example with mixed length.
        text = ("Xapien uses advanced NLP to process structured and unstructured data. This allows users to extract "
                "valuable insights from vast amounts of information. LLMs are integral to this process.")
        max_words = 10
        expected_output = [
            "Xapien uses advanced NLP to process structured and unstructured data.",
            "This allows users to extract valuable insights from vast amounts of information.",
            "LLMs are integral to this process."
        ]
        self.assertEqual(chunk_text(text, max_words), expected_output)

        # Test case 2: A sentence that is longer than max_words.
        text = "This sentence is too long and goes beyond the maximum word limit."
        max_words = 5
        expected_output = ["This sentence is too long and goes beyond the maximum word limit."]
        self.assertEqual(chunk_text(text, max_words), expected_output)

        # Test case 3: Empty text.
        text = ""
        max_words = 10
        expected_output = []
        self.assertEqual(chunk_text(text, max_words), expected_output)

        # Test case 4: A couple of sentences that are shorter than max_words.
        text = "Short sentence. Another one. And another. One more."
        max_words = 10
        expected_output = ["Short sentence. Another one. And another. One more."]
        self.assertEqual(chunk_text(text, max_words), expected_output)

    def test_re_rank_chunks(self):
        # Test case 1: general re-ranking
        chunks = [
            "Xapien uses advanced NLP to process structured and unstructured data.",
            "LLMs are integral to this process.",
            "This allows users to extract valuable insights from vast amounts of information."
        ]
        query = "How does Xapien use LLMs?"

        expected_output = [
            ("Xapien uses advanced NLP to process structured and unstructured data.", 0.94776917),
            ("LLMs are integral to this process.", 0.61207783),
            ("This allows users to extract valuable insights from vast amounts of information.", 0.021409769)
        ]

        # Check for a correct ordering on scores.
        ranked_chunks = re_rank_chunks(chunks, query)
        self.assertEqual([chunk for chunk, _ in ranked_chunks], [chunk for chunk, _ in expected_output])

        # Test case 2: one chunk re-ranking
        chunks = [
            "Xapien uses advanced NLP to process structured and unstructured data."
        ]
        query = "How does Xapien use LLMs?"

        expected_output = [
            ("Xapien uses advanced NLP to process structured and unstructured data.", 1),
        ]

        # Check for a correct ordering on scores.
        ranked_chunks = re_rank_chunks(chunks, query)
        self.assertEqual([chunk for chunk, _ in ranked_chunks], [chunk for chunk, _ in expected_output])

        # Test case 3: empty chunks re-ranking
        chunks = []
        query = "How does Xapien use LLMs?"

        expected_output = []

        # Check for a correct ordering on scores.
        ranked_chunks = re_rank_chunks(chunks, query)
        self.assertEqual([chunk for chunk, _ in ranked_chunks], [chunk for chunk, _ in expected_output])

        # Test case 4: empty query
        chunks = [
            "Xapien uses advanced NLP to process structured and unstructured data.",
            "LLMs are integral to this process.",
            "This allows users to extract valuable insights from vast amounts of information."
        ]
        query = ""

        expected_output = []

        # Check for a correct ordering on scores.
        ranked_chunks = re_rank_chunks(chunks, query)
        self.assertEqual([chunk for chunk, _ in ranked_chunks], [chunk for chunk, _ in expected_output])


if __name__ == "__main__":
    unittest.main()
