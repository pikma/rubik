'''A model that batches calls to its `predict` method.'''
import dataclasses

from typing import Optional, Tuple
import asyncio
import contextlib

import numpy as np


@dataclasses.dataclass
class _Batch:
    features: np.ndarray
    next_feature_ix: int
    ready_to_read: asyncio.Event = dataclasses.field(
        default_factory=lambda: asyncio.Event())

    predictions: Optional[np.ndarray] = None


class BatchedModel:
    def __init__(self, model, batch_size: int, feature_shape: Tuple[int]):
        self._model = model
        self._batch_size = batch_size
        self._feature_shape = feature_shape

        self._batch = self._new_batch()  # Guarded by the lock.

        self._lock = asyncio.Lock()

        self._prediction_task = asyncio.create_task(
            self._run_predictions_loop(), name="prediction_loop")

    async def __aenter__(self):
        return self

    async def __aexit__(self, type, value, traceback):
        await self.join()

    async def join(self):
        self._prediction_task.cancel()
        try:
            await self._prediction_task
        except asyncio.CancelledError:
            return

    def _new_batch(self) -> _Batch:
        return _Batch(
            features=np.ndarray((self._batch_size, ) + self._feature_shape),
            next_feature_ix=0)

    async def predict(self, x: np.ndarray):
        if x.shape != self._feature_shape:
            raise ValueError("Invalid shape for x, expected {}, got {}".format(
                self._feature_shape, x.shape))
        batch = None
        feature_ix = None
        batch_is_full = False

        async with self._lock:
            batch = self._batch
            feature_ix = batch.next_feature_ix
            batch.features[feature_ix:] = x
            batch.next_feature_ix += 1
            if batch.next_feature_ix == self._batch_size:
                self._batch = self._new_batch()
                batch_is_full = True

        if batch_is_full:
            batch.predictions = self._model.predict(
                batch.features, batch_size=self._batch_size)
            batch.ready_to_read.set()
        else:
            await batch.ready_to_read.wait()

        return batch.predictions[feature_ix, :]

    async def _run_predictions_loop(self):
        # We run predictions periodically, even if the batch is not full.
        # Otherwise, the last calls to `predict` would wait forever for the
        # batch to be full.
        while True:
            await asyncio.sleep(delay=1.0)
            old_batch = None
            async with self._lock:
                old_batch = self._batch
                self._batch = self._new_batch()

            old_batch.predictions = self._model.predict(
                old_batch.features, batch_size=self._batch_size)
            old_batch.ready_to_read.set()


contextlib.AbstractAsyncContextManager.register(BatchedModel)
