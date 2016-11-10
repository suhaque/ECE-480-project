# coding: utf-8
import random as ran
import os
from xml.etree import ElementTree
import numpy as np
import math
import threading
import regex as re
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

class Part(object):
    'component object'
    def __init__(self, partType, infoID=0, partNum=0, mft=0, mrt=0, label=0, state=['0'], KofN=1, reliability = 0):
        self.partType = partType
        self.infoID = infoID
        self.partNum = partNum
        self.mft = mft
        self.mrt = mrt
        self.label = label
        self.state = state
        self.KofN = KofN
        self.reliability = reliability
        self.children = []
        self.number = 0
    def append(self, child):
        self.children.append(child)
    def setNum(self, num):
        self.number = num
    def copy(self):
        P = Part(self.partType, self.infoID, self.partNum, self.mft, self.mrt, self.label, self.state, self.KofN, self.reliability)
        P.children = self.children
        return P

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
        n = 0
        while cycles[0]:#training cycles
            n += 1
            #if n%20000 == 0:
                #print('train cycle ' + str(n))
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
        return
    def search(self, state=0): # get fix path from state
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
    def fromStr(self, string):
        data = re.split('\n',string)
        data1 = re.split(' ',data[0])
        data2 = re.split(',',data[1])
        data3 = re.split(',',data[2])
        self.alpha = float(data1[0])
        self.gamma = float(data1[1])
        self.count = int(data1[2])
        self.reward = np.zeros((self.count,self.count))
        self.q_value = np.zeros((self.count,self.count))
        for i in range(self.count):
            if i == 63:
                print('here')
            for j in range(self.count):
                self.q_value[i,j] = float(data2[self.count*i+j])
                self.reward[i,j] = float(data3[self.count*i+j])
    def toStr(self):
        shape = self.q_value.shape
        out = str(self.alpha)+' '+str(self.gamma)+' '+str(self.count)+'\n'
        nums = []
        for i in self.q_value:
            for j in i:
                nums.append(str(j))
        out += ','.join(nums)+'\n'
        shape = self.reward.shape
        nums = []
        for i in self.reward:
            for j in i:
                nums.append(str(j))
        out += ','.join(nums)
        return out




class maintenance(object): # main class
    'Overall Maintenance class'
    def __init__(self): # constructor
        # initialize stuff
        self.parts = None
        self.Tree = None
        self.partcount = None
        self.stateCount = None
        self.currentState = ['0']
        self.structureMap = []
        self.train = [True]
        self.fixTime = None
        self.maxTime = None
        self.states = None
        self.rewards = None
        self.qnetwork = None
        self.q_thread = None
        self.file = None
    def saveQ(self, filename):
        self.train[0] = False
        self.q_thread.join()
        self.file = open(filename,'w')
        self.file.write(self.qnetwork.toStr())
        self.file.close()
        self.train[0] = True
        self.q_thread = threading.Thread(target=self.qnetwork.train,args=(self.train,))
        self.q_thread.start()
    def loadQ(self, filename):
        self.train[0] = False
        self.q_thread.join()
        self.file = open(filename,'r')
        string = self.file.read()
        self.qnetwork.fromStr(string)
        self.file.close()
        self.train[0] = True
        self.q_thread = threading.Thread(target=self.qnetwork.train,args=(self.train,))
        self.q_thread.start()
    def setStructure(self,Filename): # get structure file to parse
        root = ElementTree.parse(Filename) # parse xml file
        components = root.findall('components/part') # get components
        structure = root.find('structure/part') # get structure
        parts = root.findall("structure//*[@type='part']") # get all structure parts
        self.stateCount = len(parts) # number of parts in structure
        # not including series and parallel groupings
        self.__genPartList(components, self.currentState) # generate part list
        self.Tree = self.__genTree(structure) # generate part tree
        #self.__genStructureMap(parts) # generate structure mapping array
        self.states = [0 for j in range(2**self.stateCount)] # initialize reliability array
        for i in range(2**self.stateCount): # increment through all states
            #self.currentState[0] = bin(i)[2:][::-1] # convert state to binary
            self.currentState[0] = bin(i)[2:][::-1] # convert state to binary
            self.states[i] = self.__calcStates(self.Tree) # calculate reliability for current state
            #print(self.currentState[0],self.states[i])
        self.__genRewards() # generate reward matrix
        self.qnetwork = QNetwork()
        self.qnetwork.init_network(self.rewards)
        self.q_thread = threading.Thread(target=self.qnetwork.train,args=(self.train,))
        self.q_thread.start()

    def convert2Label(self,A): # convert part ID to part Label

        A = A.tolist()
        A[0] = bin(int(A[0]))[2:][::-1]
        for i in range(1,len(A)-1):
            j = self.structureMap[int(A[i])]
            A[i] = (int(A[i]),self.parts[j].label)
        return A
    def getFix(self,IDs):
        self.train[0] = False
        self.q_thread.join()
        broken = ['1' for i in range(self.stateCount)]
        for I in IDs:
            broken[I] = '0'
        state = int('0b'+''.join(broken[::-1]),2)
        fix = self.qnetwork.search(state)
        self.train[0] = True
        self.q_thread = threading.Thread(target=self.qnetwork.train,args=(self.train,))
        self.q_thread.start()
        return self.convert2Label(fix)
    def printParts(self):
        out1 = ""
        out2 = ""
        for i in range(self.stateCount):
            out1 += '{0:<10}'.format(i)
            out2 += '{0:<10}'.format(self.structureMap[i])
        print out1
        print out2
    def __genRewards(self): # generate reward matrix
        self.rewards = np.array([[np.nan for j in range(2**self.stateCount)] for i in range(2**self.stateCount)])
        for i in range(2**self.stateCount):
            for j in range(self.stateCount):
                temp = i^2**j
                self.rewards[i,temp] = self.states[temp] - self.states[i] - self.parts[self.structureMap[j]].mrt/self.maxTime
        self.rewards[2**self.stateCount-1][2**self.stateCount-1] = 1

    def __genTree(self, part, count=[0]):
        if part.attrib['type']=='part': # if part
            i = int(part.attrib['infoID'])
            P = self.parts[i].copy()
            P.setNum(count[0])
            self.structureMap.append(i)
            count[0] += 1
            return P
        elif part.attrib['type']=='series': # if series
            P = Part(partType='series')
            parts = part.findall('part')
            for P1 in parts:
                P.append(self.__genTree(P1, count))
            return P
        elif part.attrib['type']=='parallel': # if parallel
            P = Part(partType='parallel', KofN = int(part.attrib['k']))
            parts = part.findall('part')
            for P1 in parts:
                P.append(self.__genTree(P1, count))
            return P



    def __genPartList(self, components, state): # generate list of parts
        partCount = len(components)
        self.partcount = partCount
        partlist = []
        self.fixTime = 0
        self.maxTime = 0
        #def __init__(self, partType, infoID=0, mft=0, mrt=0, label=0, state=['0'], KofN=1):
        for part in components:
            partlist.append(Part(partType='part',
                                 infoID=int(part.attrib['infoID']),
                                 mft=float(part.attrib['mft']),
                                 mrt=float(part.attrib['mrt']),
                                 label=part.text,
                                 state=state))
            if float(part.attrib['mrt']) > self.maxTime:
                self.maxTime = float(part.attrib['mrt'])
            self.fixTime += float(part.attrib['mrt'])
        for part in partlist:
            part.reliability = math.exp(-self.fixTime/part.mft)
        self.parts = partlist


    def __calcStates(self,part): # calculate state reliabilities 
        if part.partType == 'part': # if part
            if part.number < len(self.currentState[0]):
                if self.currentState[0][part.number] == '1':
                    return part.reliability
                else:
                    return 0.0
            else:
                return 0.0
        elif part.partType == 'series': # if series
            reliability = 1.0
            for p in part.children:
                reliability *= self.__calcStates(p)
                if reliability == 0:
                    break
            return reliability
        elif part.partType == 'parallel': # if parallel
            k = part.KofN
            n = len(part.children)
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
                                if state[j] == '1':
                                    reliab *= self.__calcStates(part.children[j])
                                else:
                                    reliab *= 1.0-self.__calcStates(part.children[j])
                            else:
                                reliab *= 1.0-self.__calcStates(part.children[j])
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
                                if state[j] == '1':
                                    reliab *= self.__calcStates(part.children[j])
                                else:
                                    reliab *= 1.0-self.__calcStates(part.children[j])
                            else:
                                reliab *= 1.0-self.__calcStates(part.children[j])
                        reliability += reliab
            return reliability


        else: print 'Error ****** Unknown part type ******'
def printPrompt():
    print '{0:*^50}'.format('Accepted Commands')
    print '{0:<50}\n\t{1:<50}\n\t{2:<50}'.format('EXIT', 'Close Program', 'EXAMPLE: EXIT')
    print '{0:<50}\n\t{1:<50}\n\t{2:<50}'.format('FIX <Part_ID> <Part_ID> ...', 'Calculates best fix order', 'EXAMPLE: FIX 8 4 2 4 6 ...')
    print '{0:<50}\n\t{1:<50}\n\t{2:<50}'.format('GEN <location\\filename.txt>', 'Generates new system from xml file', 'EXAMPLE: GEN \'C:\\Users\\Username\\Documents\\Project\\filename.txt\'')
    print '{0:<50}\n\t{1:<50}\n\t{2:<50}'.format('HELP','Show this help screen', 'EXAMPLE: HELP')
    print '{0:<50}\n\t{1:<50}\n\t{2:<50}'.format('LOAD <location\\filename.dat>', 'Load trained system', 'EXAMPLE: LOAD \'C:\\Users\\Username\\Documents\\Project\\filename.dat\'')
    print '{0:<50}\n\t{1:<50}\n\t{2:<50}'.format('PARTS', 'Lists system parts and IDs', 'EXAMPLE: PARTS')
    print '{0:<50}\n\t{1:<50}\n\t{2:<50}'.format('SAVE <location\\filename.dat>', 'Save trained system', 'EXAMPLE: SAVE \'C:\\Users\\Username\\Documents\\Project\\filename.dat\'')
maint = None
run = True
printPrompt()
while(run):
    try:
        In = raw_input('Input >> ')
        In = re.findall("'.+'|^\w+$|(?<=\s)\w+$|^\w+(?=\s)|(?<=\s)\w+(?=\s)",In)
        if len(In) == 0:
            pass
        elif In[0].upper() == 'EXIT':
            if maint != None:
                maint.saveQ('autosave.dat')
            run = False
        elif In[0].upper() == 'FIX':
            if maint == None:
                raise Exception('System not loaded')
            elif len(In) > 1:
                IDs = []
                for I in In[1:]:
                    if maint.stateCount > int(I):
                        IDs.append(int(I))
                    else:
                        raise Exception('Component Id does not exist')
                print maint.getFix(IDs)
            else:
                raise Exception('FIX must have at least 1 part_ID')
        elif In[0].upper() == 'GEN':
            if len(In) == 2:
                maint = maintenance()
                maint.setStructure(In[1][1:-1])
            else:
                raise Exception('GEN only takes 2 arguments')
        elif In[0].upper() == 'HELP':
            printPrompt()
        elif In[0].upper() == 'LOAD':
            if maint == None:
                raise Exception('System not loaded')
            maint.loadQ(In[1][1:-1])
        elif In[0].upper() == 'PARTS':
            if maint == None:
                raise Exception('System not loaded')
            maint.printParts()
        elif In[0].upper() == 'SAVE':
            if maint == None:
                raise Exception('System not loaded')
            maint.saveQ(In[1][1:-1])
        else:
            print('{0:^150}'.format('Not a Valid Command'))
    except Exception as inst:
        print inst
    finally:
        if run == False and maint != None:
            maint.train[0] = False
            if not maint.file.closed:
                maint.file.close()
    #test = maintenance() # create maintenance object
    #test.setStructure('test.xml') # load structure

    #test.getFix([0,1,2])
