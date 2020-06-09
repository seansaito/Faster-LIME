import unittest

from experiments.utils.datasets import get_dataset


class TestAbstractExplainer(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_unsupported_dataset(self):
        """
        Make sure that an unimplemented abstract explainer subclass does not get instantiated
        """
        with self.assertRaises(KeyError):
            try:
                _ = get_dataset('na-dataset', {})
            except KeyError as e:
                msg = e.args[0]
                self.assertEqual(msg,
                                 'Dataset {} not found or is not supported'.format('na-dataset'))
                raise KeyError(msg)


if __name__ == '__main__':
    unittest.main()
