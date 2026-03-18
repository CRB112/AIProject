import random
import os

numStates = 10
maxEmissions = 10

def readInput(filename):
    lines = []
    try:
        if (os.path.getsize(filename) > 0):
            with open(filename, 'r') as f:
                states = {}
                start_name = None
                for line in f:
                    line = line.strip()
                    if line.startswith("start"):
                        start_name = line.split(':', 1)[1].strip()
                    else:
                        name, weights, emissions = line.split(':', 2)
                        name = name.strip()
                        weightsParsed = [float(w.strip()) for w in weights.strip('[]').split(',')]
                        emissionsParsed = [float(e.strip()) for e in emissions.strip('[]').split(',')]
                        states[name] = {'transitions' : weightsParsed, 'emissions' : emissionsParsed}
                
                #If start  was never specified, 
                #sets them to the first and last states
                if not start_name:
                    start_name = list(states.keys())[0]
                
                #If transition entries are left part-empty or too many entries
                #Were input
                for s in states.keys():
                    while len(states[s]['transitions']) < len(states.keys()):
                        states[s]['transitions'].append(0)
                    while len(states[s]['transitions']) > len(states.keys()):
                        states[s]['transitions'].pop()

                states = trimEmissions(states)

                return (start_name, states)
        else:
            generateInput(filename, numStates, maxEmissions)
            return readInput(filename)
    except FileNotFoundError:
        generateInput("Input.txt", numStates, maxEmissions)
        return readInput("Input.txt")

def generateInput(filename, numS, maxE):
    with open(filename, 'a') as f:
        for i in range(numS):
            ranTrans = [random.random() for _ in range(numS)]
            totalTrans = sum(ranTrans)
            ranTrans = [v / totalTrans for v in ranTrans]

            ranEm = [random.random() for _ in range(maxE)]  # ensure >0
            totalEm = sum(ranEm)
            ranEm = [v / totalEm for v in ranEm]

            # Print normalized emissions
            print(f"Generated emissions: {ranEm}")

            f.write("s" + str(i) + 
                    ":[" + 
                    ','.join(str(v) for v in ranTrans) + 
                    ']:[' + 
                    ','.join(str(v) for v in ranEm) + ']\n')

def trimEmissions(statesD):
    maxEm = 0
    for s in statesD.keys():
        val = len(statesD[s]['emissions'])
        if val > maxEm:
            maxEm = val
    for s in statesD.keys():
        while len(statesD[s]['emissions']) < maxEm:
            statesD[s]['emissions'].append(0)
        while len(statesD[s]['emissions']) > maxEm:
            statesD[s]['emissions'].pop()

    return statesD
    