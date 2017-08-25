from stingray.utils import simon
import numpy as np

def execute_parallel(work, list_of_operations, *args, **kwargs):
	"""
	This is the starting point of executing in parallel:
	This function should be called when there are lines of code that
	should be executed in parallel.
	These lines of code should be wrapped in a function and passed here as the "work" argument.
	*args are the arguments that will be passed to the work function and they should consist of slicable objects.

	What execute_parallel should do is keep calling the execute_functions found in prefered_parallel_libraries one by one
	with the work function and its *args. If the execute_function called returns the "uninstalled" global variable, it will
	be assumed that the intended distributed library is not installed so the execute_parallel will keep looking for an installed library.
	If it runs out of execute functions, it will call _execute_sequential.
	
	list_of_operations is a list containing each after processing operation to be done on each item in the returned items.

	"""
	for library_name, execute_fn in prefered_parallel_libraries.items():
		try:
			status_or_values = execute_fn(work, list_of_operations, *args, **kwargs)
		except Exception as e:
			error_message = ""
			if hasattr(e, 'message'):
				error_message = (e.message)
			else:
				error_message = (e)
			simon("A problem occured while computing in parallel mode. ("+ str(error_message)+ ") Switching to sequential..")
			return _execute_sequential(work,*args)
		#continue looking
		if( status_or_values is uninstalled ):
			continue
		else:
			return status_or_values

	simon("Did not find any distributed computing libraries installed, executing sequentially...")
	return _execute_sequential(work,*args)

"""
What every _execute method should do is: try to import the intended library.
If not installed, it should return "uninstalled" (global variable).
If it is installed, it should import it. Slice the arguments depending on how many processes
will be created. Run the work function and then combine their returned values and return them as a one big combined value.

"""

def _execute_dask(work, list_of_operations, *args, **kwargs):
	"""
	This function works as follow:
	1- Check how many cores are availble
	2- Determine each process slice size.
	3- Loop over the arguments and slice them according to each process share.
	4- After setting the process share append the function call to the list of tasks.
	5- When we are done filling our tasks, we should call for dask's compute and wait for the results.
	6- The results that are returned from computing all the tasks is a collection (specifically tuple)
	   of the returned values by each process.
	   Example:
		   arr = [1,2,3,4,5,6,7,8,9,10]
		   def add(arr):
		   		return np.asarray(list).sum()
		   def avg(arr):
		   		return add(arr) / len(arr)
		   def avg_sum (arr):
			   	sum = 0
			   	for number in arr:
					sum += arr
				avg = sum / len(arr)
				return sum, avg
			
			_execute_dask(avg_sum, [add, avg], arr)


		Assume that avg_sum is our work function and we would like to execute this distributedly using dask.
		_execute_dask will be called by the following arguments: work = avg_sum, args = [arr]

		Assuming that the number of cores available are 2. Each process share would be equal to len(arr)/2 .. = 5.
		We slice the arguments (arr) accordingly, process1 share = [1, .. ,5]   process2 share = [6, .. ,10] 
		We then compute.
		The expected returned values are ((15,3), (40,8))
		We go ahead and create two lists of each attribute first = [15, 40] second = [3, 8]
		We call each consequent after processing method given in list_of_operations.

		return list_of_operations[0](first), list_of_operations[1](second)

		that will give us the (45,5.5) we were looking for.

		The expected keywords are:
		jit = True/False, Which refers wether or not to use 'numba.jit'. numba.jit should be used when the work function consists
		of raw python code. Something like looping over arrays and calculating their sum.
		Default True.

		shared_res = True/False, This refers to wether or not there is an object being shared between all threads that all threads
		could read and write in it. Default: False.
		Example:

		my_arr = []

		def my_work(intervals):
			
			beginning_of_my_interval = intervals[0]
			my_arr.append(beginning_of_my_interval)

		This work functions not only does it read the my_arr, but also each thread needs to modify it. This is when shared_res should equal to True.

	"""
	
	try:
		from multiprocessing import cpu_count
		from dask import compute, delayed
		import dask.multiprocessing
		import dask.threaded
		if(kwargs.get('jit') == None or kwargs['jit'] == True):
			from numba import jit
	except Exception as e: 
		return uninstalled

	processes_count = cpu_count() if kwargs.get("cpus") == None else kwargs["cpus"]
	tasks = []
	intervals = args[0]
	for i in range(processes_count):
            
            process_share = int( len(intervals) / processes_count )
            starting_index = i * process_share
           
            if(i == processes_count -1):
            	#last process takes from the starting index till the end of the array
            	ending_index = len(intervals)
            else:
            	ending_index = min((starting_index + process_share), len(intervals))

            #slice each argument
            process_args = []
            for argument in args:
                sliced_argument = argument[starting_index:ending_index]
                process_args.append( sliced_argument )

            if(ending_index > starting_index):
            	if(kwargs.get('jit') == False):
            		# not using jit
            		tasks.append(delayed(work)(*process_args))
            	else:
            		# using jit
            		tasks.append(delayed(jit(work))(*process_args))

     # Default: Use Multiprocessing
	_get = dask.multiprocessing.get

    # If there is a shared res, use threaded
	if(kwargs.get('shared_res') == True):
		_get = dask.threaded.get
	
	list_of_results = list( compute(*tasks, get = _get) )

	return _post_processing(list_of_results, list_of_operations)



def _execute_multiprocess(work, list_of_operations, *args, **kwargs):
	try:
		from multiprocessing import Process, cpu_count, Queue
		from threading import Thread 
	except:
		# It will never return uninstalled.
		return uninstalled

	processes_count = cpu_count()
	processes = []
	intervals = args[0]

	communication_channels = []
	for i in range(processes_count):
            
            process_share = ( len(intervals) // processes_count )
            starting_index = i * process_share
           
            if(i == processes_count -1):
            	#last process takes from the starting index till the end of the array
            	ending_index = len(intervals)
            else:
            	ending_index = min((starting_index + process_share), len(intervals))

            #slice each argument
            process_args = []
            for argument in args:
                sliced_argument = argument[starting_index:ending_index]
                process_args.append( sliced_argument )

			#append on the arguments the channel which will be used to communicate a subprocess with the main thread.           
            communication_que = Queue()
            process_args.append(communication_que)
            process_args.append(i)  # for the subprocess to know its index
            if(ending_index > starting_index):
                if(kwargs.get('jit') == False):
                    # not using jit
                    process = Process(target = (work), args = process_args)
                else:
                    # using jit
                    process = Process(target = (jit(work)), args = process_args)
                
                processes.append(process)
                communication_channels.append(communication_que)

	def recvv(que, i ):
		while True:
				if(not que.empty()):
					results[i] = que.get()
					break

    # initiate computing processes
	for process in processes:
		process.start()

	results = [[] for _ in range(len(processes))]
	recv_threads = []
	
			

	# Create Recieving threads
	for i, process in enumerate(processes):
		t = Thread(target = recvv, args = (communication_channels[i],i,))
		recv_threads.append(t)
		t.start()

	# join threads and processes
	for i, process in enumerate(processes):
		process.join() 	
		recv_threads[i].join()

	#Check if there was any Exceptions
	for result in results:
		if(isinstance(result, Exception)):
			raise result
	return  _post_processing(results,list_of_operations)


def _execute_sequential(work, *args):
	return work(*args)


def _post_processing(listOfResults,list_of_operations):
	"""
	we consider the listOfResults to be in the form of : [(Attribute1, Attribute2, Attribute3),
							  							  (Attribute1, Attribute2, Attribute3)]
	"""
	listOfAttributes = []
	try:
		# more than one object is being returned by a single work function.
		number_of_attributes = len(listOfResults[0])

		for i in range(number_of_attributes):
			attribute = []
			for result in listOfResults:
				attribute.append(result[i])
			listOfAttributes.append(attribute)

				# listOfAttributes.append(item[i] for item in listOfResults)
	except TypeError:
		# only a single item is being returned by a single work function. 
		listOfAttributes.append(listOfResults)
	
	final_values = []
	for i, attribute in enumerate(listOfAttributes):
		try:
			final_values.append(list_of_operations[i](attribute))
		except IndexError:
			simon("ran out of operations in post processing.. the rest of the attributes will be gathered using the default operation" \
				  "(Summation).	")
			final_values.append(post_add(attribute))

	return tuple(final_values) if len(final_values) != 1 else final_values[0]

# Some basic post processing methods

def post_add (arr):
	sum = arr[0]
	for element in arr[1:]:
		sum += element
	return sum


def post_concat_arrays(list_of_arrays):
	big_array = np.empty(0)
	numpy_lists = False

	if(len(list_of_arrays) > 0):
		if( type(list_of_arrays[0]) == type(np.empty(0)) ):
			numpy_lists = True

	for array in list_of_arrays:
		big_array = np.concatenate([big_array, array])


	return big_array if numpy_lists else list(big_array)

from collections import OrderedDict
prefered_parallel_libraries = OrderedDict()
prefered_parallel_libraries["multiP"] = _execute_multiprocess
prefered_parallel_libraries["dask"] = _execute_dask

uninstalled = object()

