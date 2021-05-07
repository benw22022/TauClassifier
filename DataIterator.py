"""
Class for a lazy iterator
"""

import uproot
import numba

class DataIterator:

	def __init__(self, file_list, var_list, batch_size, step_size='10000 MB', cuts=""):
		self._lazy_array = uproot.lazy(file_list, filter_name=var_list, step_size=batch_size, cuts=cuts,)

		self._batch_size = batch_size
		self._idx = 0
		self._len = len(self._lazy_array)

	@staticmethod
	@numba.jit(nopython=True)
	def _get_batch(lazy_array, idx, batch_size):
		return lazy_array[idx * batch_size: idx * batch_size + batch_size]

	def __next__(self):
		if self._idx + self._batch_size < self._len:
			#batch = self._lazy_array[self._idx * self._batch_size : self._idx * self._batch_size + self._batch_size]
			batch = self._get_batch(self._lazy_array, self._idx, self._batch_size)
			self._idx += 1
			return batch

	def reset(self):
		self._idx = 0


if __name__ == "__main__":
	pass