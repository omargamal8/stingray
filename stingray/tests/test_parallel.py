from stingray.parallel import *
import numpy as np

class TestParallel:
	def setup_class(self):
		self.interval = np.arange(-10,11,1)
		
	def test_single_return(self):
		def work(arr):
			sum = 0
			for element in arr:
				sum += element
			return sum

		returned = execute_parallel(work, [post_add], self.interval)
		assert returned == np.sum(self.interval)
		

	def test_multiple_returns(self):

		def work(arr):
			sum = 0
			mul = 1
			for element in arr:
				sum += element
				mul *= element

			return sum, mul

		def post_mul(arr):
			mul = 1
			for element in arr:
				mul *= element
			return mul
		
		index = np.where(self.interval == 0 )
		no_zeros = np.delete( self.interval, index)
		returned = execute_parallel(work, [post_add, post_mul], no_zeros)

		assert returned == (np.sum(self.interval), post_mul(no_zeros))


	def test_multiple_returns_arrays(self):

		def work(arr):
			return [1], [2]

		length = 10
		returned = execute_parallel(work, [post_concat_arrays, post_concat_arrays], np.arange(length))

		a = np.asarray([1 for _ in range(length)])
		b = np.asarray([2 for _ in range(length)])

		assert np.allclose(returned[0], a)
		assert np.allclose(returned[1], b)

	


