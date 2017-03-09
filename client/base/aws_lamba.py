import json
import threading

import boto3


class FitnessCall(threading.Thread):

    def __init__(self, gene):
        super().__init__()
        self.client = boto3.client('lambda')
        self.gene = gene
        self.result = None

    def call(self, payload_dict):
        response = self.client.invoke(
            FunctionName='test-function-001',
            InvocationType='RequestResponse',
            LogType='None',
            Payload=json.dumps({"json": json.dumps(payload_dict)}).encode('utf-8')
        )

        return response['Payload'].read()

    def run(self):
        w = {
            'weights': self.gene
        }

        self.result = self.call(w)

    def __call__(self):
        self.start()

    def get_fitness(self):
        self.join()
        return self.result



if __name__ == '__main__':
    import random, math

    calls = [FitnessCall([random.normalvariate(0, 1) for i in range(340)]) for i in range(50)]

    [c() for c in calls]

    for c in calls:
        print(math.sqrt(int(c.get_fitness())))
