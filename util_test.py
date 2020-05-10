'''Tests for util.py'''

import numpy as np
import unittest

import util


class FakeModel:
    def predict(self, features, batch_size):
        return features


class CubeTest(unittest.TestCase):
    '''Tests for ModelBatcher.'''
    def test_requests_of_one(self):
        batcher = util.ModelBatcher(batch_size=3,
                                    model=FakeModel(),
                                    feature_shape=(1, ),
                                    feature_dtype='int')
        batcher.enqueue_predictions(np.asarray([[1]]), '1')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.enqueue_predictions(np.asarray([[2]]), '2')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.enqueue_predictions(np.asarray([[3]]), '3')
        self.assertListEqual(list(batcher.get_predictions()), [
            (np.asarray([[1]]), '1'),
            (np.asarray([[2]]), '2'),
            (np.asarray([[3]]), '3'),
        ])

        batcher.enqueue_predictions(np.asarray([[4]]), '4')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.flush()
        self.assertListEqual(list(batcher.get_predictions()), [
            (np.asarray([[4]]), '4'),
        ])

    def test_requests_of_one_and_two(self):
        batcher = util.ModelBatcher(batch_size=3,
                                    model=FakeModel(),
                                    feature_shape=(1, ),
                                    feature_dtype='int')
        batcher.enqueue_predictions(np.asarray([[1], [2]]), '1')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.enqueue_predictions(np.asarray([[3]]), '3')

        results = list(batcher.get_predictions())
        self.assertEqual(len(results), 2)
        self.assertTrue((results[0][0] == [[1], [2]]).all())
        self.assertEqual(results[0][1], '1')
        self.assertEqual(results[1], (np.asarray([[3]]), '3'))

        batcher.enqueue_predictions(np.asarray([[4]]), '4')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.flush()
        self.assertListEqual(list(batcher.get_predictions()), [
            (np.asarray([[4]]), '4'),
        ])

    def test_request_split_between_batches(self):
        batcher = util.ModelBatcher(batch_size=3,
                                    model=FakeModel(),
                                    feature_shape=(1, ),
                                    feature_dtype='int')
        batcher.enqueue_predictions(np.asarray([[1]]), '1')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.enqueue_predictions(np.asarray([[2]]), '2')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.enqueue_predictions(np.asarray([[3], [4]]), '3')
        self.assertListEqual(list(batcher.get_predictions()), [
            (np.asarray([[1]]), '1'),
            (np.asarray([[2]]), '2'),
        ])

        batcher.enqueue_predictions(np.asarray([[5]]), '5')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.flush()

        results = list(batcher.get_predictions())
        self.assertEqual(len(results), 2)
        self.assertTrue((results[0][0] == [[3], [4]]).all())
        self.assertEqual(results[0][1], '3')
        self.assertEqual(results[1], (np.asarray([[5]]), '5'))

    def test_request_bigger_than_batch_size(self):
        batcher = util.ModelBatcher(batch_size=2,
                                    model=FakeModel(),
                                    feature_shape=(1, ))
        first_features = np.arange(1, 6).reshape((5, 1))
        batcher.enqueue_predictions(first_features, '1')
        self.assertFalse(list(batcher.get_predictions()))
        batcher.enqueue_predictions(np.asarray([[6]]), '6')
        results = list(batcher.get_predictions())

        self.assertEqual(len(results), 2)
        self.assertTrue((results[0][0] == first_features).all())
        self.assertEqual(results[0][1], '1')
        self.assertEqual(results[1], (np.asarray([[6]]), '6'))


if __name__ == '__main__':
    unittest.main()
