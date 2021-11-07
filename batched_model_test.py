'''Tests for batched_model.py.'''

import asyncio
import unittest

import numpy as np

import batched_model


class BatchedModelTest(unittest.IsolatedAsyncioTestCase):
    '''Tests for the BatchedModel class.'''

    async def test_concurrent_request(self):
        NUM_REQUESTS = 1000

        class FakeModel:
            def predict(self, x, batch_size):
                assert x.shape == (64, 2)
                assert batch_size == 64
                return np.matmul(x, np.array([[2], [1]]))

        async with batched_model.BatchedModel(
                FakeModel(), batch_size=64, feature_shape=(2,)) as model:
            async def request(i: int):
                return await model.predict(np.array([i, 3*i]))

            results = await asyncio.gather(
                *(asyncio.create_task(request(i),
                                      name='request_{}'.format(i))
                  for i in range(NUM_REQUESTS)))

        # The result should be 2*x + 1*(3*x) = 5*x.
        self.assertListEqual(results, [5*i for i in range(NUM_REQUESTS)])


if __name__ == '__main__':
    unittest.main()
