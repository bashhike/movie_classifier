import unittest
from movie_classifier import inference


class TestMovieClassifier(unittest.TestCase):
    """
    Testing the basic cases for the classifier.
    """

    def test_comedy(self):
        self.assertEqual(
            inference("comedy",
                      "A writer returns to his hometown where he faces the childhood nemesis whose life he ultimately "
                      "ruined, only the bully wants to relive their painful past by torturing him once again.",
                      "bilstm.pth")['genre'],
            "Comedy"
        )

    def test_drama(self):
        self.assertEqual(
            inference("Cry, the Beloved Country",
                      "His wife having recently died, Thomas Jefferson accepts the post of United States ambassador "
                      "to pre-revolutionary France, though he finds it difficult to adjust to life  in a country "
                      "where the aristocracy subjugates an increasingly restless peasantry. In Paris, he becomes "
                      "smitten with cultured artist Maria Cosway, but, when his daughter visits from Virginia "
                      "accompanied by her attractive slave, Sally Hemings, Jefferson's attentions are diverted.",
                      "bilstm.pth")['genre'],
            "Drama"
        )


if __name__ == '__main__':
    unittest.main()
