import json

from xor import XOR


def handler(params, lambda_):
    # get genotype from args
    json_str = params.get('json', None)
    json_data = json.loads(json_str)

    # build phenotype
    xor_solver = XOR(json_data['weights'])

    # evaluate
    fitness = xor_solver.evaluate()

    return fitness

if __name__ == '__main__':
    import random

    w = {'weights': [random.random() * 0.1 - 0.2 for i in range(12)]}

    p = {
        'json': json.dumps(w)
    }

    print(handler(p, None))
