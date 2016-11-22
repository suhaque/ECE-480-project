# coding: utf-8
import random as ran
import os
from xml.etree import ElementTree
import numpy as np
import math
import threading
import re
import scipy.sparse as sps
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
    def isActive(self, state):
        if self.partType == 'part':
            if self.number < len(state[0]):
                return state[0][self.number] == '1'
            return False
        if self.partType == 'series':
            for child in self.children:
                if not child.isActive(state):
                    return False
            return True
        if self.partType == 'parallel':
            count = 0
            for child in self.children:
                if child.isActive(state):
                    count += 1
            if count >= self.KofN:
                return True
            return False
    def copy(self):
        P = Part(self.partType, self.infoID, self.partNum, self.mft, self.mrt, self.label, self.state, self.KofN, self.reliability)
        P.children = self.children
        return P

class QNetwork(object): # qnetwork class
    'Qnetwork class'
    def __init__(self, alpha=0.9, gamma=1.0): # constructor, alpha = learn rate, gamma = memory
        self.alpha = alpha # set alpha
        self.gamma = gamma # set gamma
        self.count = 1 # initialize to something, I just chose 1
        self.reward = None # initialize numpy arrays for reward and q_value matrix
        self.q_value = None
        self.state = 0 # set state to 0

    def init_network(self, reward, state=0): # initialize network
        self.count = reward.shape[0] # number of states
        self.reward = reward # set reward matrix
        self.q_value = sps.lil_matrix(reward.shape) # set Q value matrix
        self.state = state # set initial state

    def train(self, cycles, mrts, maxtime, maxRel):
        n = 0
        time = 0
        while cycles[0]:#training cycles
            n += 1
            state = self.state #set current state
            r = ran.randint(1, 100) #get random number from 1-100
            if r > 30: # 70% of the time
                actions = np.argsort(self.q_value[state, :].toarray(),kind='mergesort')[0][::-1]
                action = actions[self.reward[state, actions].toarray(0)[0] != 0][0]
                # increment Q value  -  multiline
                if state != action: # test if next state is the same as the current state
                    self.q_value[state, action] = self.q_value[state, action] + \
                    self.alpha*(self.reward[state, action] + \
                    self.gamma*np.max(self.q_value[action, :].toarray()[0]) - \
                    self.q_value[state, action])
                    part = int(math.log(action ^ state, 2))
                    if time < maxtime:
                        time += mrts[part]
                else:
                    self.q_value[state, action] = self.q_value[state, action] + \
                    self.alpha*(self.reward[state, action] + maxRel*(maxtime - time)  + \
                    self.gamma*np.max(self.q_value[action, :].toarray()[0]) - \
                    self.q_value[state, action])
                    r = ran.randint(1, 100) #get random number from 1-100
                    if r > 10:
                        action = ran.randint(0, self.count-1)
                    else:
                        action = 0
                    time = 0
                self.state = action# set next state
            else: # 30% of the time
                indices = np.nonzero(self.reward[state, :].toarray()[0])[0]
                if len(indices) == 0:
                    state = ran.randint(0, self.count-1)
                    continue
                action = ran.randint(0, len(indices)-1) # get random action
                action = indices[action]
                # increment Q value
                if state != action: # test if next state is the same as the current state
                    self.q_value[state, action] = self.q_value[state, action] + \
                    self.alpha*(self.reward[state, action] + \
                    self.gamma*np.max(self.q_value[action, :].toarray()[0]) - \
                    self.q_value[state, action])
                    part = int(math.log(action ^ state, 2))
                    if time < maxtime:
                        time += mrts[part]
                else:
                    self.q_value[state, action] = self.q_value[state, action] + \
                    self.alpha*(self.reward[state, action] + maxRel*(maxtime - time)  + \
                    self.gamma*np.max(self.q_value[action, :].toarray()[0]) - \
                    self.q_value[state, action])
                    r = ran.randint(1, 100) #get random number from 1-100
                    if r > 10:
                        action = ran.randint(0, self.count-1)
                    else:
                        action = 0
                    time = 0
                # set next state
                self.state = action
        return
    def search(self, maxtime, mrts, maxRel, state=0): # get fix path from state
        time = 0
        path = [] # component path
        states = [state] # state path
        #cost = 0 # initial cost
        limit = int(math.log(self.count, 2))
        for i in range(limit+1): # loop up to number of states
            # get the sorted indices of the row of the q matrix for the current state and reverse it
            test1 = self.q_value[state, :].toarray()[0]
            test2 = self.reward[state, :].toarray()[0]
            actions = np.argsort(self.q_value[state, :].toarray(),kind='mergesort')[0][::-1]
            action = actions[self.reward[state, actions].toarray(0)[0] != 0][0]

            if state == action:
                return [path, states]
            part = int(math.log(action ^ state, 2))
            path.append(part)
            states.append(action)
            # if next state is the same as current state found end of path
            # return path and cost
            # from state and action
            # calcuate repaired part ID
            state = action # set next action

            # iterate through actions
            # increment Q value
            #self.q_value[state, action] = self.q_value[state, action] + \
            #self.alpha*(self.reward[state, action] + \
            #self.gamma*np.max(self.q_value[action, :].toarray()[0]) - \
            #self.q_value[state, action])
            #cost += self.reward[state, action] # add to cost
 #           if state != action: # test if next state is the same as the current state
 #               self.q_value[state, action] = self.q_value[state, action] + \
 #               self.alpha*(self.reward[state, action] + \
 #               self.gamma*np.max(self.q_value[action, :].toarray()[0]) - \
 #               self.q_value[state, action])
 #               part = int(math.log(action ^ state, 2))
 #               path.append(part)
 #               states.append(action)
 #               if time < maxtime:
 #                   time += mrts[part]
 #           else:
 #               self.q_value[state, action] = self.q_value[state, action] + \
 #               self.alpha*(self.reward[state, action] + maxRel*(maxtime - time)  + \
 #               self.gamma*np.max(self.q_value[action, :].toarray()[0]) - \
 #               self.q_value[state, action])
 #               return [path, states]
            
        return None # if no path found return None
    #print q matrix
    def fromStr(self, string):
        string = re.split('\n', string)
        info = re.split(',', string[0])
        alpha = float(info[0])
        gamma = float(info[1])
        count = int(info[2])
        q_value_nonzero = np.array(re.split(',', string[1])).astype(float)
        q_value_r = np.array(re.split(',', string[2])).astype(int)
        q_value_c = np.array(re.split(',', string[3])).astype(int)
        reward_nonzero = np.array(re.split(',', string[4])).astype(float)
        reward_r = np.array(re.split(',', string[5])).astype(int)
        reward_c = np.array(re.split(',', string[6])).astype(int)

        self.alpha = alpha
        self.gamma = gamma
        self.count = count
        self.reward = sps.coo_matrix((reward_nonzero, (reward_r, reward_c)), (count, count)).tocsr()
        self.q_value = sps.coo_matrix((q_value_nonzero, (q_value_r, q_value_c)), (count, count)).tolil()

    def toStr(self):
        out = str(self.alpha)+','+str(self.gamma)+','+str(self.count)+'\n'

        q_value = self.q_value.tocoo()
        q_value_data = q_value.data
        q_value_nonzero = q_value.nonzero()
        q_value_r = q_value_nonzero[0]
        q_value_c = q_value_nonzero[1]

        out += ','.join(q_value_data.astype(str))+'\n'
        out += ','.join(q_value_r.astype(str))+'\n'
        out += ','.join(q_value_c.astype(str))+'\n'

        reward = self.reward.tocoo()
        reward_data = reward.data
        reward_nonzero = reward.nonzero()
        reward_r = reward_nonzero[0]
        reward_c = reward_nonzero[1]

        out += ','.join(reward_data.astype(str))+'\n'
        out += ','.join(reward_r.astype(str))+'\n'
        out += ','.join(reward_c.astype(str))+'\n'

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
        self.fixTime = 0
        self.maxTime = None
        self.states = None
        self.rewards = None
        self.qnetwork = None
        self.q_thread = None
        self.file = None
        self.mrts = None
    def saveQ(self, filename):
        self.train[0] = False
        self.q_thread.join()
        self.file = open(filename,'w')
        self.file.write(self.qnetwork.toStr())
        self.file.close()
        self.train[0] = True
        self.q_thread = threading.Thread(target=self.qnetwork.train,args=(self.train, self.mrts, self.fixTime, self.states[-1], ))
        self.q_thread.start()
    def loadQ(self, filename):
        self.train[0] = False
        self.q_thread.join()
        self.file = open(filename,'r')
        string = self.file.read()
        self.qnetwork.fromStr(string)
        self.file.close()
        self.train[0] = True
        self.q_thread = threading.Thread(target=self.qnetwork.train,args=(self.train, self.mrts, self.fixTime, self.states[-1], ))
        self.q_thread.start()
    def setStructure(self, Filename): # get structure file to parse
        root = ElementTree.parse(Filename) # parse xml file
        components = root.findall('components/part') # get components
        structure = root.find('structure/part') # get structure
        parts = root.findall("structure//*[@type='part']") # get all structure parts
        self.stateCount = len(parts) # number of parts in structure

        # not including series and parallel groupings
        self.__genPartList(components, self.currentState) # generate part list
        self.fixTime = 0
        for part in parts:
            i = int(part.attrib['infoID'])
            self.fixTime += self.parts[i].mrt
        self.__genRel()
        self.Tree = self.__genTree(structure) # generate part tree
        self.mrts = []
        for i in range(self.stateCount):
            self.mrts.append( self.parts[self.structureMap[i]].mrt )
        self.states = [0 for j in range(2**self.stateCount)] # initialize reliability array
        for i in range(2**self.stateCount): # increment through all states
            #self.currentState[0] = bin(i)[2:][::-1] # convert state to binary
            self.currentState[0] = bin(i)[2:][::-1] # convert state to binary
            if i == 63:
                print ''
            self.states[i] = self.__calcStates(self.Tree) # calculate reliability for current state
            #print(self.currentState[0],self.states[i])
        self.__genRewards() # generate reward matrix
        self.qnetwork = QNetwork()
        self.qnetwork.init_network(self.rewards)
        self.q_thread = threading.Thread(target=self.qnetwork.train,args=(self.train, self.mrts, self.fixTime, self.states[-1], ))
        self.q_thread.start()

    def convert2Label(self, A): # convert part ID to part Label
        output = '\n{0:^20} {1:^20} {2:^40}\n'.format('Current',
                                                    'Current State',
                                                    'Part to Fix')
        output += '{0:^20} {1:^20} ({2:^20}, {3:^20})\n\n'.format('State',
                                                               'Reliability',
                                                               'Part Id',
                                                               'Part Label')
        for i in range(len(A[0])):
            j = self.structureMap[int(A[0][i])]
            current_state = bin(int(A[1][i]))[2:][::-1]
            part_id = int(A[0][i])
            part_label = self.parts[j].label
            reliability = self.states[A[1][i]]
            output += '{0:^20} {1:^20} ({2:^20}, {3:^20})\n'.format(current_state,
                                                                    reliability,
                                                                    part_id,
                                                                    part_label)
        i = len(A[0])
        current_state = bin(int(A[1][i]))[2:][::-1]
        reliability = self.states[A[1][i]]
        output += '{0:^20} {1:^20}\n'.format(current_state, reliability)
        return output
    def getFix(self, IDs):
        self.train[0] = False
        self.q_thread.join()
        broken = ['1' for i in range(self.stateCount)]
        for I in IDs:
            broken[I] = '0'
        state = int('0b'+''.join(broken[::-1]),2)
        fix = self.qnetwork.search(self.fixTime, self.mrts, self.states[-1], state)
        self.train[0] = True
        self.q_thread = threading.Thread(target=self.qnetwork.train,args=(self.train, self.mrts, self.fixTime, self.states[-1], ))
        self.q_thread.start()
        return fix
    def evaluate(self, fix):
        parts = fix[0]
        states = fix[1]
        total_score = 0
        total_time = 0
        for i in range(len(parts)):
            reliab = self.states[states[i]]
            #reliab2 = self.states[states[i+1]]
            time = self.parts[self.structureMap[parts[i]]].mrt
            score = time * reliab
            total_time += time
            total_score += score
        remain_score = (self.fixTime - total_time) * self.states[-1]
        return (total_score + remain_score)/self.fixTime
        print 'test'
    def printParts(self):
        out1 = ""
        out2 = ""
        for i in range(self.stateCount):
            out1 += '{0:<10}'.format(i)
            out2 += '{0:<10}'.format(self.structureMap[i])
        print out1
        print out2
    def __genRewards(self): # generate reward matrixs
        rows = []
        cols = []
        data = []
        maxim = None
        for i in range(2**self.stateCount):
            for j in range(self.stateCount):
                temp = i^(2**j)
                if temp > i:
                    #val = -0.05
                    rows.append(i)
                    cols.append(temp)
                    CONST = 1
                    #val = CONST*(self.states[temp] - self.states[i]) - self.parts[self.structureMap[j]].mrt/self.maxTime
                    val = self.states[i] * self.parts[self.structureMap[j]].mrt/self.fixTime
                    if val == 0:
                        val = -0.1
                    if maxim == None or val > maxim:
                        maxim = val
                    data.append(val)
        maxim *= 10
        rows.append(2**self.stateCount-1)
        cols.append(2**self.stateCount-1)
        data.append(maxim)
        self.rewards = sps.coo_matrix((data, (rows, cols)), dtype='f')
        self.rewards = self.rewards.tocsr()
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
        self.parts = partlist
    def __genRel(self):
        for part in self.parts:
            part.reliability = math.exp(-self.fixTime/part.mft)


    def __calcStates(self, part): # calculate state reliabilities 
        if part.partType == 'part': # if part
            if part.isActive(self.currentState):
                return part.reliability
            else:
                return 0.0
        elif part.partType == 'series': # if series
            if part.isActive(self.currentState):
                reliability = 1.0
                for p in part.children:
                    reliability *= self.__calcStates(p)
                return reliability
            else:
                return 0.0
        elif part.partType == 'parallel': # if parallel
            k = part.KofN
            n = len(part.children)
            if not part.isActive(self.currentState):
                return 0.0
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
                            if reliab == 0:
                                break
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
                            if reliab == 0:
                                break
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
# Main Start
maint = None
run = True
printPrompt()
while(run):
    try:
        In = raw_input('\nInput >> ')
        In = re.findall("'.+'|^\w+$|(?<=\s)\w+$|^\w+(?=\s)|(?<=\s)\w+(?=\s)", In)
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
                fix = maint.getFix(IDs)
                score = maint.evaluate(fix)
                print 'Score: {}\n{}'.format(score, maint.convert2Label(fix))
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
            if maint.file != None:
                if not maint.file.closed:
                    maint.file.close()