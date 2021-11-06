'''Utilities.'''
import copy
import collections
from dataclasses import dataclass
import itertools

from typing import Any, Deque, Iterator, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class _RowIx:
    '''An index of a row in the ModelBatcher.

    Used both for indexing in 'feature_batches' and in '_predictions'.
    '''
    batch_ix: int
    row_ix: int

    def get_batch_ix_preceding(self):
        if self.row_ix == 0:
            return self.batch_ix - 1
        return self.batch_ix


@dataclass
class _Request:
    '''A request made to 'enqueue_prediction'.

    The 'end' index is exclusive.
    '''
    begin: _RowIx
    end: _RowIx
    request_id: Any


class ModelBatcher:
    '''Batches requests to a model's predict function.

    Pass features to 'enqueue_predictions'. Call 'get_predictions' to iterate
    through the predictions that are ready.

    'enqueue_predictions' takes multiples rows of features per call, and
    'get_predictions' will return the multiple predictions once they are all
    ready. 'get_predictions' also returns the 'request_id' passed to
    'enqueue_predictions'. This can be anything, it doesn't have to be unique
    per request, and it can be None.
    '''
    def __init__(self,
                 batch_size: int,
                 model: tf.keras.Model,
                 feature_shape=Tuple[int],
                 feature_dtype='float'):
        self._batch_size = batch_size
        self._model = model
        self._feature_shape = feature_shape
        self._feature_dtype = feature_dtype

        self._feature_batches: Deque[np.ndarray] = collections.deque()
        self._predictions: Deque[np.ndarray] = collections.deque()
        # When batches are deleted, we do not recompute the indices in
        # _requests and _next_write_ix. Therefore when accessing
        # _features_batches or _predictions using some of the indices,
        # _num_deleted_batches should be substracted from the index.
        self._next_write_ix = _RowIx(batch_ix=0, row_ix=0)
        self._num_deleted_batches = 0
        self._requests: Deque[_Request] = collections.deque()

        # Invariants:
        # _feature_batches[i] corresponds to _predictions[i]

    def enqueue_predictions(self,
                            features_array: np.ndarray,
                            request_id: Any = None):
        '''Enqueues features for predictions.

        'request_id' can be anything, it will be returned back in
        'get_predictions'. 'features_array' must be an array of shape (n, 20,
        24). The n rows will be yielded together by 'get_predictions'.
        '''

        num_rows = features_array.shape[0]
        previous_ix = copy.deepcopy(self._next_write_ix)

        num_copied = 0
        while num_copied != num_rows:
            num_copied += self._append_features_to_current_batch(
                features_array[num_copied:, :])

        self._requests.append(
            _Request(begin=previous_ix,
                     end=copy.deepcopy(self._next_write_ix),
                     request_id=request_id))

    def _append_features_to_current_batch(self, features_array):
        '''Appends features to the current batch.

        Stops after filling one batch. Returns the number of rows appended.
        '''
        batch_ix = (self._next_write_ix.batch_ix - self._num_deleted_batches)
        if batch_ix == len(self._feature_batches):
            self._feature_batches.append(
                np.ndarray(shape=(self._batch_size, ) + self._feature_shape,
                           dtype=self._feature_dtype))
        elif batch_ix > len(self._feature_batches):
            raise Exception(
                'batch_ix = {}, len(self._feature_batches) = {}'.format(
                    batch_ix, len(self._feature_batches)))

        batch = self._feature_batches[batch_ix]
        num_rows_left = self._batch_size - self._next_write_ix.row_ix
        num_to_copy = min(num_rows_left, features_array.shape[0])
        next_row_ix = self._next_write_ix.row_ix + num_to_copy
        batch[self._next_write_ix.row_ix:next_row_ix, :] = (
            features_array[:num_to_copy, :])

        if next_row_ix == self._batch_size:
            self._next_write_ix.batch_ix += 1
            self._next_write_ix.row_ix = 0
        else:
            self._next_write_ix.row_ix = next_row_ix
        return num_to_copy

    def get_predictions(self) -> Iterator[Tuple[np.ndarray, Any]]:
        '''Iterates through the predictions.

        Yields tuples (np.ndarray of predictions, request_id).

        The array of predictions has shape (n), where n is the number of rows
        that was passed to 'enqueue_predictions'.
        '''

        # Apply predictions to feature batches that are full.
        first_unprocessed_batch_ix = len(self._predictions)
        first_non_full_batch_ix = (self._next_write_ix.batch_ix -
                                   self._num_deleted_batches)
        for batch in itertools.islice(self._feature_batches,
                                      first_unprocessed_batch_ix,
                                      first_non_full_batch_ix):
            self._predictions.append(
                self._model.predict(batch, batch_size=self._batch_size))

        # Yield the requests that are ready.
        while True:
            if not self._requests:
                break

            request = self._requests[0]
            last_batch_ix = (request.end.get_batch_ix_preceding() -
                             self._num_deleted_batches)
            if last_batch_ix >= len(self._predictions):
                # Part of this request is contained in a batch that hasn't been
                # processed yet (because it is not full).
                break

            request = self._requests.popleft()

            predictions = []
            first_batch_ix = request.begin.batch_ix - self._num_deleted_batches

            for batch_ix, pred_batch in itertools.islice(
                    enumerate(self._predictions), first_batch_ix,
                    last_batch_ix + 1):
                begin_row_ix = (request.begin.row_ix
                                if batch_ix == first_batch_ix else 0)
                end_row_ix = (request.end.row_ix if batch_ix == last_batch_ix
                              and request.end.row_ix != 0 else
                              self._batch_size)
                predictions.append(pred_batch[begin_row_ix:end_row_ix])
            yield (np.concatenate(predictions), request.request_id)

        # Clear up the feature batches and the prediction batches that have
        # been processed and that are now useless.
        if self._requests:
            num_batches_to_delete = (self._requests[0].begin.batch_ix -
                                     self._num_deleted_batches)
        else:
            # All requests have been read.
            num_batches_to_delete = len(self._feature_batches)
        for _ in range(num_batches_to_delete):
            self._feature_batches.popleft()
            self._predictions.popleft()
        self._num_deleted_batches += num_batches_to_delete

    def flush(self):
        '''Forces the model to predict all the requests received so far.

        This causes the model to run on all the data received so far, even if
        it requires running a prediction on a batch smaller than 'batch_size'.

        It is important to call this function once after having requested all
        the data required, to make sure that all of it is passed to the model.
        After calling this function, calling 'get_predictions' will yield all
        the predictions, and the ModelBatcher can be discarded.
        '''

        first_unprocessed_batch_ix = len(self._predictions)
        for feature_batch in itertools.islice(self._feature_batches,
                                              first_unprocessed_batch_ix,
                                              len(self._feature_batches)):
            self._predictions.append(
                self._model.predict(feature_batch,
                                    batch_size=self._batch_size))
