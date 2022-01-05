import json
import sys

def update(updatelist):
    alternativelist={'data':updatelist}
    with open('datajson.json') as f:
        bufferlist=json.load(f)
    print(bufferlist)
    bufferlist=alternativelist
    print(bufferlist)
    with open('datajson.json', 'w') as f:
        json.dump(bufferlist,f)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   update(sys.argv[1])