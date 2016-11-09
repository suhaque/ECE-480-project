# coding: utf-8
import random as ran
import os
from xml.etree import ElementTree
import numpy as np
import math
# 1. stateet the gammaction parameter, and environment rewardstate in matrix reward.
# 2. Initialize matrix Q to zero.
# 3. For each episode:
# stateelect action random initial state.
# Do While the goal state hasn't been reached.
# stateelect one among all possible actionstate for the current state.
# Using thistate possible action, consider going to the next state.
# Get maximum Q value for thistate next state based on all possible actions.
# Compute: Q(state, action) += Alpha*(reward(state, action) +
# Gammaction * Max[Q(next state, all actions)]- Q(state, action))
# stateet the next state astate the current state.
# End Do
# End For

class QNetwork(object): # qnetwork class
    'Qnetwork class'
    def __init__(self, alpha=0.5, gamma=0.9): # constructor, alpha = learn rate, gamma = memory
        self.alpha = alpha # set alpha
        self.gamma = gamma # set gamma
        self.count = 1 # initialize to something, I just chose 1
        self.reward = np.zeros((1,1)) # initialize numpy arrays for reward and q_value matrix
        self.q_value = np.zeros((1,1))
        self.state = 0 # set state to 0

    def init_network(self, reward, state=0): # initialize network
        self.count = len(reward[0, :]) # number of states
        self.reward = reward # set reward matrix
        self.q_value = np.zeros(reward.shape) # set Q value matrix
        self.state = state # set initial state

    def train(self, cycles):
        for n in range(cycles):#training cycles
            state = self.state #set current state
            r = ran.randint(1, 100) #get random number from 1-100
            if r > 30: # 70% of the time
                actions = np.argsort(self.reward[state, :],kind='mergesort')[::-1] # get indices of self.reward sorted in descending order
                for action in actions: # iterate through indices to find
                    if not np.isnan(self.reward[state, action]): # first reward that isn't NAN
                        break
                # increment Q value  -  multiline
                self.q_value[state, action] = self.q_value[state, action] + \
                self.alpha*(self.reward[state, action] + \
                self.gamma*np.max(self.q_value[action, :]) - \
                self.q_value[state, action])
                if state == action: # test if next state is the same as the current state
                    self.state = ran.randint(0, self.count-1)
                self.state = action# set next state
            else: # 30% of the time
                action = ran.randint(0, self.count-1) # get random action
                cap = 0 # initialize loop counter
                while np.isnan(self.reward[state, action]) and cap < 1000:# get random with reward != NAN
                    action = ran.randint(0, self.count-1)
                    cap += 1
                if cap == 1000: # if no path exists, change state
                    state = ran.randint(0, self.count-1)
                    continue
                # increment Q value
                self.q_value[state, action] = self.q_value[state, action] + \
                self.alpha*(self.reward[state, action] + \
                self.gamma*np.max(self.q_value[action, :]) - \
                self.q_value[state, action])
                # set next state
                self.state = action
    def search(self,state=0): # get fix path from state
        path = [state] # initial state
        cost = 0 # initial cost
        for i in range(self.count): # loop up to number of states
            # get the sorted indices of the row of the q matrix for the current state and reverse it
            actions = np.argsort(self.q_value[state, :],kind='mergesort')[::-1]
            for action in actions: # iterate through actions
                if not np.isnan(self.reward[state, action]): # if reward for action is not NAN
                    # increment Q value
                    self.q_value[state, action] = self.q_value[state, action] + \
                    self.alpha*(self.reward[state, action] + \
                    self.gamma*np.max(self.q_value[action, :]) - \
                    self.q_value[state, action])
                    cost += self.reward[state, action] # add to cost
                    if state == action:# if next state is the same as current state
                    # found end of path
                        return np.concatenate((path, [cost])) # return path and cost
                    # from state and action
                    # calcuate repaired part ID
                    temp = int(math.log(action ^ state)/math.log(2)) 
                    path.append(temp) # append part ID
                    state = action # set next action
                    break
        return None # if no path found return None
    #print q matrix
    def print_q(self):
        print str(self.q_value) + '\n'




class maintenance(object): # main class
    'Overall Maintenance class'
    def __init__(self): # constructor
        # initialize stuff
        self.parts = 0
        self.partcount = 0
        self.stateCount = 0
        self.currentState = '0b0'
        self.structureMap = 0
        self.fixTime = 0
        self.maxTime = 0
        self.states = 0
        self.rewards = 0

    def getStructure(self,Filename): # get structure file to parse
        root = ElementTree.parse(Filename) # parse xml file
        components = root.findall('components/part') # get components
        structure = root.find('structure/part') # get structure
        parts = root.findall("structure//*[@type='part']") # get all structure parts
        self.stateCount = len(parts) # number of parts in structure
        # not including series and parallel groupings
        self.__genPartList(components) # generate part list
        self.__genStructureMap(parts) # generate structure mapping array
        self.states = [0 for j in range(2**self.stateCount)] # initialize reliability array
        for i in range(2**self.stateCount): # increment through all states
            self.currentState = bin(i)[2:][::-1] # convert state to binary
            self.states[i] = self.__calcStates(structure) # calculate reliability for current state
            #print(self.currentState,self.states[i])
        self.__genRewards() # generate reward matrix

    def convert2Label(self,A): # convert part ID to part Label
        A = A.tolist()
        A[0] = bin(int(A[0]))[2:][::-1]
        for i in range(1,len(A)-1):
            j = self.structureMap[int(A[i])]
            A[i] = self.parts[j][3]
        return A

    def __genRewards(self): # generate reward matrix
        self.rewards = np.array([[np.nan for j in range(2**self.stateCount)] for i in range(2**self.stateCount)])
        for i in range(2**self.stateCount):
            for j in range(self.stateCount):
                temp = i^2**j
                self.rewards[i,temp] = self.states[temp] - self.states[i] - self.parts[self.structureMap[j]][2]/self.maxTime
        self.rewards[2**self.stateCount-1][2**self.stateCount-1] = 1

    def __genStructureMap(self, components): # generate structure array
        partCount = len(components)
        self.partcount = partCount
        partlist = [0 for j in range(partCount)]
        for part in components:
            i = int(part.attrib['partNum'])
            partlist[i] = int(part.attrib['infoID'])
        self.structureMap = partlist

    def __genPartList(self, components): # generate list of parts
        partCount = len(components)
        self.partcount = partCount
        partlist = [[0 for i in range(4)] for j in range(partCount)]
        self.fixTime = 0
        self.maxTime = 0
        for part in components:
            i = int(part.attrib['infoID'])
            partlist[i][1] = float(part.attrib['mft'])
            partlist[i][2] = float(part.attrib['mrt'])
            if partlist[i][2] > self.maxTime:
                self.maxTime = partlist[i][2]
            self.fixTime += partlist[i][2]
            partlist[i][3] = part.text
        for part in components:
            i = int(part.attrib['infoID'])
            partlist[i][0] = math.exp(-self.fixTime/partlist[i][1])
        self.parts = partlist


    def __calcStates(self,part): # calculate state reliabilities 
        if part.attrib['type']=='part': # if part
            i = int(part.attrib['infoID'])
            j = int(part.attrib['partNum'])
            if j < len(self.currentState):
                if self.currentState[j] == '1':
                    return self.parts[i][0]
                else:
                    return 0.0
            else:
                return 0.0
        elif part.attrib['type']=='series': # if series
            reliability = 1.0
            parts = part.findall('part')
            for p in parts:
                reliability *= self.__calcStates(p)
                if reliability == 0:
                    break
            return reliability
        elif part.attrib['type']=='parallel': # if parallel
            k = int(part.attrib['k'])
            parts = part.findall('part')
            n = len(parts)
            if n-k > k:
                reliability = 0.0
                for i in range(2**n-1):
                    reliab = 1.0
                    state = bin(i)[2:][::-1]
                    active = 0
                    for L in range(len(state)):
                        active += int(state[L])                   
                    if active < k:
                        for j in range(n):
                            if j < len(state):
                                if state[j]=='1':
                                    reliab *= self.__calcStates(parts[j])
                                else:
                                    reliab *= 1.0-self.__calcStates(parts[j])
                            else:
                                reliab *= 1.0-self.__calcStates(parts[j])
                        #print(state,reliab)
                        reliability += reliab
                reliability = 1-reliability
            else:
                reliability = 0.0
                for i in range(2**n-1,-1,-1):
                    reliab = 1.0
                    state = bin(i)[2:][::-1]
                    active = 0
                    for L in range(len(state)):
                        active += int(state[L])                   
                    if active >= k:
                        for j in range(n):
                            if j < len(state):
                                if state[j]=='1':
                                    reliab *= self.__calcStates(parts[j])
                                else:
                                    reliab *= 1.0-self.__calcStates(parts[j])
                            else:
                                reliab *= 1.0-self.__calcStates(parts[j])
                        #print(state,reliab)
                        reliability += reliab
            return reliability


        else: print 'Error ****** Unknown part type ******'

test = maintenance() # create maintenance object
test.getStructure('test.xml') # load structure

N = len(test.rewards) # set testing size

A = QNetwork() # create qnetwork object
A.init_network(test.rewards) # input reward matrix

print 'start training'
A.train(1000) # train for 1000 cycles
print 'finished training'
#A.print_q()
for i in range(N): # output for all states
    print  test.convert2Label(A.search(i))
