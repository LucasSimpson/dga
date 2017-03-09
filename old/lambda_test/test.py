import boto3
import threading

client = boto3.client('lambda')


def call(n):
    response = client.invoke(
        FunctionName='test-function-001',
        InvocationType='RequestResponse',
        LogType='None',
        Payload=b'{"number": %s}' % n
    )

    return response['Payload'].read()


class Test(threading.Thread):
    def __init__(self, n):
        threading.Thread.__init__(self)
        self.n = n

    def run(self):
        print '%s: %s' % (self.n, call(self.n))


threads = [Test(i) for i in range(100)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print('done')
