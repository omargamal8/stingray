from stingray.parallel import execute_parallel, post_add, post_concat_arrays
import numpy as np
import warnings
from astropy.tests.helper import pytest
from stingray import AveragedCrossspectrum, Lightcurve

def single_return_work(arr, que = None, index = 0):
	sum = 0
	for element in arr:
		sum += element
	if(que != None):
		que.put(sum)
	else:
		return sum

class TestMultiP:
	def setup_class(self):
		self.interval = np.arange(-10,11,1)
		
	def test_single_return(self):
		interval = self.interval
		parallel_library = "multiP"			
		with warnings.catch_warnings(record=True) as w:
			returned = execute_parallel(single_return_work, [post_add], interval, prefered=parallel_library)
			assert returned == np.sum(interval)
			# Check that it was actually executed in parallel not sequential.
			for warning in w:
				assert not("switching to sequential" in str(warning.message))
		


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

