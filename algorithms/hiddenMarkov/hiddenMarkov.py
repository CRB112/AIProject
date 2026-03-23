import readInput
import random

maxObservations = 10

#Returns an observation sequence [] based on start start and
#Emission probabilities
def getObsSeq(startState, statesD, maxO):
    states = list(statesD.keys())
    seq = []

    for _ in range(maxO):
        state = startState if len(seq) == 0 else random.choice(states)
        em = statesD[state]['emissions']

        seq.append(random.choices(range(len(em)), weights=em, k=1)[0])
    return seq

def algo(startState, statesD):
    obs = getObsSeq(startState, statesD, maxObservations)
    states = list(statesD.keys())
    numObs = len(obs)

    V = [{} for _ in range(numObs)] # Setting up blank table
    path = {} # settings up blank path
    
    #Settings first state to 'startState'
    for state in states:
        if state == startState:
            V[0][state] = 1
        else:
            V[0][state] = 0
        path[state] = [state]

    print("Start state is " + str(startState) + "\n")
    #Main algorithm
    #takes steps starting at 1 (startState already established)
    #creates a new path
    #Checks each state and saves the probabililty of reaching the current state
    #Multiplies transition probability x emission probability x previous vertibi probability
    #If the probability of traversing through all of these states to the current has the highest
    #Probability values, selects the previous state that maximmizes probability and extends the path
    for step in range(1, numObs):
        newPath = {}

        for curr in states:
            maxProb = -1
            prevStateSel = None
            emProb = statesD[curr]['emissions'][obs[step]]

            for prevState in states:
                prob = V[step-1][prevState] * statesD[prevState]['transitions'][states.index(curr)] * emProb
                if prob > maxProb:
                    maxProb = prob
                    prevStateSel = prevState

            V[step][curr] = maxProb
            newPath[curr] = path[prevStateSel] + [curr]

        #New path setting for next iteration and prints some data
        path = newPath
        bestCurr = max(V[step], key=V[step].get)
        print(f"Step {step}, Observation: {obs[step]}")
        print(f"Best state: {bestCurr}, prob = {V[step][bestCurr]}")
        print(f"Current best path: {path[bestCurr]}")
        print("======Next Step======\n")

    #The algorithm has reached the last step and grabs the emission probability
    #From the previous step to the finish
    maxProb = -1
    lastState = None
    for state in states:
        if V[-1][state] > maxProb:
            maxProb = V[-1][state]
            lastState = state
        
    bestPath = path[lastState]

    print("Observations -> " + str(obs))
    print("Best Path    -> " + str(bestPath))

if __name__ == "__main__":
    start, states = readInput.readInput('Input.txt')

    algo(start, states)

