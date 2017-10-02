import sys
from itertools import permutations
global_var = {
        'min_supp': float(sys.argv[2]),
        'min_conf': float(sys.argv[4]),
        'frequent_items': {}
        }
class node:
    """A tree node"""
    def __init__(self, name, left = None, right = None, top = None):
        self.num = 1 if name is not 'root' else 0
        self.name = name
        self.child = []
        self.left = left
        self.right = right
        self.top = top
    def add_child(self, node):
        self.child.append(node)
        print('add ' + node.name + ' to ' + self.name)
    def add_num(self):
        self.num += 1
    def show(self):
        print([c.name for c in self.child])
        print([c.num for c in self.child])
    def findpath(self, nodeName, path):
        if self.child:
            for item in self.child:
                if item.name == nodeName:
                    path.append(item)
                else:
                    item.findpath(nodeName, path)
    def toroot(self):
        path = self.name
        while self.top.name != 'root':
            path = self.top.name + ' ' + path
            self = self.top
        return path

def add_node(transaction, n, connection_table):
    if transaction:
        item =  transaction.pop(0)
        for c in n.child:
            if c.name == item:
                c.add_num()
                add_node(transaction, c, connection_table)
                return
        if item in connection_table: 
            newNode = node(item, left = connection_table[item], top = n)
            connection_table[item].right = newNode
        else:
            newNode = node(item, top = n)
        connection_table[item] = newNode
        n.add_child(newNode)
        add_node(transaction, n.child[-1], connection_table)
        return
def build_tree(nameOfroot, data, headtable):
    root = node(nameOfroot)
    connection_table = {}
    frequent_items = sorted([k for k,v in headtable.items() if v >= global_var['min_supp']], key = lambda e :headtable[e])
    for index, items in enumerate(data):
        items = [item for item in items if item in frequent_items]
        data[index] = sorted(items, reverse = True, key = lambda e : global_var['headtable'][e])
        add_node(sorted(items, reverse = True, key = lambda e : global_var['headtable'][e]), root, connection_table)
    return root, frequent_items
def build_headtable(data):
    headtable = {}
    for items in data:
        for item in items:
            if item in headtable:
                headtable[item] += 1
            else:
                headtable[item] = 1
    return headtable
def mining(tree, frequent_item, posfix):
        tmp = posfix
        for item in frequent_item:
            path = []
            tree.findpath(item, path)
            data = []
            for node in path:
                data += node.num * [node.toroot()]
            data = [items.split() for items in data]
            headtable = build_headtable(data)
            sub_tree, sub_frequent_item = build_tree('root', data, headtable)
            sub_frequent_item.remove(item)
            if len(sub_frequent_item) == 0:
                if tmp != '':
                    global_var['frequent_items'][item + ' ' + tmp] = headtable[item]
            else:
                if tmp != '':
                    mining(sub_tree, sub_frequent_item, item + " " + tmp)
                else:
                    mining(sub_tree, sub_frequent_item, item + tmp)

def FP_groth():
    with open('input.txt', 'r') as f:
        rawData = f.read()    
        rawData = rawData.split('\n')
        del rawData[-1]
        global_var['min_supp'] = int(len(rawData) * global_var['min_supp'])
        data = [items.split() for items in rawData]
    print(data)
    global_var['headtable'] = build_headtable(data)
    F_tree, frequent_item = build_tree('root', data, global_var['headtable'])
    for item in frequent_item:
        global_var['frequent_items'][item] = global_var['headtable'][item]
        mining(F_tree, [item], '')
        
FP_groth()
print(global_var['frequent_items'])

with open('output.txt','w') as f:
    for k, v in global_var['frequent_items'].items():
        tmp = k.split()
        if len(tmp) > 2:
            for p in permutations(tmp): 
                print(p)
                for num_item in range(1, len(tmp)):
                    key = ' '.join(list(p)[:num_item])
                    if key in global_var['frequent_items']:
                        conf = v / global_var['frequent_items'][key]
                        if conf >= global_var['min_conf']:
                            f.write( '{ '+ ','.join(list(p)[:num_item]) +' }' + ' ---> { ' + ','.join(list(p)[num_item:]) + ' } (' + str(conf) + ')'+'\n')
