from stingray.utils import simon




def execute_parallel(work, *args):
	"""
	This is the starting point of executing in parallel:
	This function should be called when there are lines of code that
	should be executed in parallel.
	These lines of code should be wrapped in a function and passed here as the "work" argument.
	*args are the arguments should be consisting of slicable objects.

	What execute_parallel should do keep calling the execute_functions found prefered_parallel_libraries one by one
	and passes to it the work function and its *args. If the execute_function called returns the "uninstalled" global variable,
	it will keep looking for an installed library. If it runs out of execute functions, it will call _execute_sequential.

	"""
	for library_name, execute_fn in prefered_parallel_libraries.items():
		status_or_values = _execute_dask(work,*args)
		if( status_or_values is uninstalled ):
			continue
		else:
			return status_or_values

	simon("Did not find any distributed computing libraries installed, executing sequentially...")
	return _execute_sequential(work,*args)


def _execute_dask(work,*args):
	"""
	What every _execute method should do is: try to import the intended library.
	If not installed, it should return "uninstalled" (global variable).
	If is installed, it should import and run the work function and then combine
	and return the values returned from the work function.
	
	"""
	
	try:
		from multiprocessing import cpu_count
		from dask import compute, delayed
		import dask.multiprocessing
	except:
		return uninstalled

	processes_count = cpu_count()
	tasks = []
	intervals = args[0]
	for i in range(processes_count):
            
            process_share = int( len(intervals) / processes_count )
            starting_index = i * process_share
            ending_index = min((starting_index + process_share), len(intervals))

            #slice each argument
            process_args = []
            for argument in args:
            	process_args.append( argument[starting_index:ending_index] )

            tasks.append(delayed(work)(*process_args))

	list_of_results = list( compute(*tasks, get = dask.multiprocessing.get) )

	summation = list_of_results[0]

	for single_result in list_of_results[1:]:
		try:
			for i, item in enumerate(single_result):
				summation[i] += item
		except:
			# only a single non-iteratable object is returned by the work function
			summation += single_result
	print(summation)
	return summation

prefered_parallel_libraries = {"dask":_execute_dask}
uninstalled = object()
