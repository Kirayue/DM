from copy import deepcopy
import sys
from itertools import permutations
result = {}
def PreprocessData(filename):
    data = {}
    with open(filename, 'r') as f:
        rawData = f.read()
        rawData = rawData.split('\n')
        del rawData[-1]
        for sequence in rawData:
            tmp = sequence.split() #tmp for row rawData split
            SID = tmp.pop(0)
            it = iter(tmp)
            for time in it:
                item = next(it)
                if item in data:
                    data[item]['SID'].append(SID)
                    data[item]['EID'].append((int(time),))
                else:
                    data[item] = {}
                    data[item]['SID'] = [SID]
                    data[item]['EID'] = [(int(time),)]
    return data
def pruneData(data, min_sup):
    tmp = deepcopy(data)
    with open(sys.argv[4],'a') as f:
        for key, val in tmp.items():
            if len(set(val['SID'])) < min_sup:
                del data[key]
            else:
                f.write(key + " SUP:" + str(len(set(val['SID']))) + '\n')
                print(key, len(set(val['SID'])))
    return data
def iteratorSet(setA):
    for index, value in enumerate(setA['SID']):
        yield value, setA['EID'][index]
def intersect(setA, setB, mode):
    if mode is 0:
        intersection = {'SID': [], 'EID': []}
        for SIDA, EIDA in iteratorSet(setA):
            for SIDB, EIDB in iteratorSet(setB):
                if SIDA is SIDB and EIDA[-1] < EIDB[-1]:
                    intersection['EID'].append(EIDA + (EIDB[-1],))
                    intersection['SID'].append(SIDA)
        return intersection
    elif mode is 1:
        intersection = {'SID': [], 'EID': []}
        for SIDA, EIDA in iteratorSet(setA):
            for SIDB, EIDB in iteratorSet(setB):
                if SIDA is SIDB and EIDA[-1] is EIDB[0]:
                    intersection['EID'].append(EIDA + EIDB)
                    intersection['SID'].append(SIDA)
        return intersection
if  __name__ == '__main__':
    min_sup = float(sys.argv[2]) 
    n = 1
    data = PreprocessData('seqdata.dat')
    result['1-Sequences'] = pruneData(data, min_sup)
    while(result[ str(n) + '-Sequences']):
        result[ str(n + 1) + '-Sequences'] = {}
        for p1 in list(result[ str(n) + '-Sequences'].keys()):
            for p2 in list(result['1-Sequences'].keys()):
                if p1[-1] is not p2:
                    middle = p1.split('||')
                    tmp = middle.pop().split('  ')
                    tmp.append(p2)
                    tmp = list(set(tmp))
                    middle.append(('  ').join(sorted(tmp)))
                    middle = ('||').join(middle)
                    if middle[(-3) * n + 2:] in result[str(n) + '-Sequences'] and middle[:-3] in result[str(n) + '-Sequences']:
                        intersection = intersect(result[str(n) + '-Sequences'][middle[:-3]], result[str(n) + '-Sequences'][middle[(-3) * n + 2:]], 1)
                        if len(intersection['SID']) is not 0:
                            result[str(n + 1) + '-Sequences'][middle] = intersection
                    if  (p1 + '||' + p2)[(-3) * n + 2:] in result[str(n) + '-Sequences']:
                        sub = (p1 + '||' + p2)[(-3) * n + 2:] 
                        intersection = intersect(result[str(n) + '-Sequences'][p1], result[str(n) + '-Sequences'][sub], 0)
                        if len(intersection['SID']) is not 0:
                            result[str(n + 1) + '-Sequences'][p1 + '||' + p2] = intersection
        result[ str(n + 1) + '-Sequences'] = pruneData(result[str(n + 1) + '-Sequences'], min_sup)
        n = n + 1
