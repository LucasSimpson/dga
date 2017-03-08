

def handler(kwargs, lambda_):
	val = kwargs.get('number', 0)

	return val**2 + 2*val - 10

	