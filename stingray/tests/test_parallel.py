from stingray.parallel import *
import numpy as np
import warnings
from astropy.tests.helper import pytest
from stingray import AveragedCrossspectrum, Lightcurve

# parallel_library = None

# def next_parallel_library_gen():
# 	parallel_libraries = ["multiP", "dask"]
# 	i = 0
# 	while True:
# 		yield parallel_libraries[i]
# 		i+=1
# 		i = i % len(parallel_libraries)

# global_generator = next_parallel_library_gen()

class TestMultiP:
	def setup_class(self):
		self.interval = np.arange(-10,11,1)
		# self.parallel_library = next(global_generator)
		# self.parallel_library = "multiP"
		
	def test_single_return(self):
		with warnings.catch_warnings(record=True) as w:
			
			def work(arr, que = None, index = 0):
				sum = 0
				for element in arr:
					sum += element
				if(que != None):
					que.put(sum)
				else:
					return sum

			returned = execute_parallel(work, [post_add], self.interval)
			assert returned == np.sum(self.interval)
			# Check that it was actually executed in parallel not sequential.
			for warning in w:
				assert not("switching to sequential" in str(warning.message))
		

	def test_multiple_returns(self):

		with warnings.catch_warnings(record=True) as w:
			def work(arr, que = None, index = 0):
				sum = 0
				mul = 1
				for element in arr:
					sum += element
					mul *= element

				if(que != None):
					que.put([sum, mul])
				else:
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
			# Check that it was actually executed in parallel not sequential.
			for warning in w:
				assert not ("switching to sequential" in str(warning.message))




	def test_multiple_returns_arrays(self):



		with warnings.catch_warnings(record=True) as w:
			def work(arr, que = None, index = 0):
				a = []
				b = []
				for _ in arr:
					a+=[1]
					b+=[2]
				if(que != None):
					que.put( [ a, b ] )
				else:
					return a, b
			returned = execute_parallel(work, [post_concat_arrays, post_concat_arrays], self.interval)
			
			a,b = work(self.interval)
			
			assert np.allclose(returned[0], a)
			assert np.allclose(returned[1], b)

			# Check that it was actually executed in parallel not sequential.
			for warning in w:
				assert not ("switching to sequential" in str(warning.message))


	def test_switch_to_sequential(self):

		def work(arr, que = None, index = 0):
			return None

		with warnings.catch_warnings(record=True) as w:
			execute_parallel(work, [lambda arr: arr], 2)
			assert any("switching to sequential" in str(warning.message) for warning in w)

	def test_exposing_exception(self):
		def work(arr, que = None, index = 0):
			if(que != None):
				que.put(ValueError)
			else:
				raise ValueError


		with pytest.raises(ValueError) as ex:
			execute_parallel(work, [lambda arr: arr], self.interval)


	def test_AvCs_parallel(self):
		tstart = 0.0
		tend = 20.0
		dt = np.longdouble(0.0001)

		time = np.arange(tstart + 0.5*dt, tend + 0.5*dt, dt)

		counts1 = np.random.poisson(0.01, size=time.shape[0])
		counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])

		lc1 = Lightcurve(time, counts1, gti=[[tstart, tend]], dt=dt)
		lc2 = Lightcurve(time, counts2, gti=[[tstart, tend]], dt=dt)

		av_cs_seq = AveragedCrossspectrum(lc1, lc2, segment_size=1, parallel=False)
		av_cs_parallel = None
		with warnings.catch_warnings(record=True) as w:
			av_cs_parallel = AveragedCrossspectrum(lc1, lc2, segment_size=1, parallel=True)
			assert not any("switching to sequential" in str(warning.message) for warning in w)

		for cs_seq, cs_parallel in zip(av_cs_seq.cs_all, av_cs_parallel.cs_all):
			assert cs_seq.df == cs_parallel.df
			assert np.allclose(cs_seq.freq, cs_parallel.freq)
			assert np.allclose(cs_seq.lc1.time, cs_parallel.lc1.time)
			assert np.allclose(cs_seq.lc2.time, cs_parallel.lc2.time)
			assert np.allclose(cs_seq.lc1.counts, cs_parallel.lc1.counts)
			assert np.allclose(cs_seq.lc2.counts, cs_parallel.lc2.counts)

	def test_rebin_parallel(self):
		dt = 0.03125
		rebinning_factor = 0.5
		lc_size =  (10 ** 4)
		final_element = dt * lc_size
		times = np.arange(0,final_element,dt)
		counts = np.random.rand(lc_size) * 100
		lc1 = Lightcurve(times, counts)

		lc_rebinned_seq = lc1.rebin(dt+rebinning_factor, parallel = False)
		lc_rebinned_parallel = None

		with warnings.catch_warnings(record=True) as w:
			lc_rebinned_parallel = lc1.rebin(dt+rebinning_factor, parallel = True)
			assert not any("switching to sequential" in str(warning.message) for warning in w)
		assert np.allclose(lc_rebinned_seq.time, lc_rebinned_parallel.time)
		assert np.allclose(lc_rebinned_seq.counts, lc_rebinned_parallel.counts)

# If Dask is uninstalled it will automatically switch to MultiProcessing and pass all Dask's tests
# class TestDask(TestMultiP):
# 	pass