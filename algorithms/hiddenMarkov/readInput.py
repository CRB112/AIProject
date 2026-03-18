import random

numStates = 5
maxEmissions = 5

def readInput(filename):
    lines = []
    try:
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

            start_state = states[start_name]
            return (start_state, states)
    except FileNotFoundError:
        generateInput("Input.txt")
        return readInput("Input.txt")

def generateInput(filename, numS, maxE):
    with open(filename, 'a') as f:
        for i in range(numS):
            ranTrans = [random.random() for _ in range(numS)]
            ranEm = [random.random() for _ in range(random.randint(1, maxE))]

            totalTrans = sum(ranTrans)
            totalEm = sum(ranEm)

            ranTrans = [v / totalTrans for v in ranTrans]
            ranEm = [v / totalEm for v in ranEm]

            f.write("s" + str(i) + 
                    ":[" + 
                    ','.join(str(v) for v in ranTrans) + 
                    ']:[' + 
                    ','.join(str(v) for v in ranEm) + ']\n')

    