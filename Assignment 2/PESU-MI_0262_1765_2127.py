import heapq

def Reverse(lst):
    lst.reverse()
    return lst

def A_star_Traversal(cost, heuristic, start_point, goal):
    pq =[]
    par=[]
    d=[]
    dest = set()
    nodes = len(cost[start_point])
    for _ in range(1,nodes+2):
        d.append(1e9 + 7)
        par.append(start_point)
    for node in goal:
        dest.add(node)
    pq.append([heuristic[start_point],start_point])
    d[start_point]=0
    while(len(pq)!=0):
        u = heapq.heappop(pq)
        if(d[u[1]]+heuristic[u[1]]<u[0]):
            continue
        #print(u[1])
        for v in range(1,nodes):
            if(cost[u[1]][v]!=-1 and (d[v]>(d[u[1]]+cost[u[1]][v]))):
                #print(u[1])
                d[v]=d[u[1]]+cost[u[1]][v]
                par[v]=u[1]
                heapq.heappush(pq,[d[v]+heuristic[v],v])
    ans=[]
    dist= 1e9 + 1
    dest_node =-1
    l = []
    #for i in range(1,len(heuristic)):
     #   print(d[i],i)
    for node in goal:
        if(dist > d[node]):
            dest_node = node
            dist = d[node]
    v=dest_node
    if dest_node==-1:
        return l
    ans.append(v)
    while(par[v]!=start_point):    
        ans.append(par[v])
        v=par[v]
    ans.append(start_point)
    #print(Reverse(ans))
    l = Reverse(ans)
	
    return l

def UCS_Traversal(cost, heuristic, start_point, goal):
    pq =[]
    par=[]
    d=[]
    nodes = len(cost[start_point])
    for _ in range(1,nodes+2):
        d.append(1e9 + 7)
        par.append(start_point)
    pq.append([0,start_point])
    d[start_point]=0
    while(len(pq)!=0):
        u = heapq.heappop(pq)
        if(d[u[1]]<u[0]):
            continue
        #print(u[1])
        for v in range(1,nodes):
            #print(d[v]>(d[u[1]]+cost[u[1]][v]))
            if(cost[u[1]][v]!=-1 and (d[v]>(d[u[1]]+cost[u[1]][v]))):
                d[v]=d[u[1]]+cost[u[1]][v]
                par[v]=u[1]
                heapq.heappush(pq,[d[v],v])
    ans = []
    dist= 1e9 + 1
    dest_node =-1
    #for i in range(1,len(heuristic)):
     #   print(d[i],i)
    for node in goal:
        if(dist > d[node]):
            dest_node = node
            dist = d[node]
    v=dest_node
    l = []
    if(dest_node==-1):
        return l
    ans.append(v)
    while(par[v]!=start_point):    
        ans.append(par[v])
        v=par[v]
    ans.append(start_point)
    #print(Reverse(ans))
    l = Reverse(ans)
	
    return l

T = 0
def DFS_Traversal(cost, heuristic, start_point, goal):
    nodes = len(cost[start_point])
    visited = set()
    dest_node = -1
    u = start_point
    global T
    T = 0
    t=[]
    par=[]
    for _ in range(0, nodes+1):
        t.append(1e9 +7)
        par.append(start_point)
    def dfs(u):
        #print(u)
        global T
        t[u] = T
        T=T+1
        visited.add(u)
        for v in range(1,nodes):
            if cost[u][v]!=-1 and v not in visited:
                par[v]=u
                #print(u,v)
                dfs(v)

    dfs(u)
    tm = 1e9 + 1
    l = []
    for i in goal:
        #print(t[i])
        if tm > t[i]:
            dest_node = i
            tm = t[i]
            #print(i,t[i])
	#print("DFS")
	#print(dest)
    if dest_node==-1:
        return l
    v = dest_node
    ans=[v]
    while(par[v]!=start_point):    
        ans.append(par[v])
        v=par[v]
    ans.append(start_point)
	#print(Reverse(ans))	
    l = Reverse(ans)

    return l


'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):
    l = []
    #T = 0
    t1 = DFS_Traversal(cost, heuristic, start_point, goals)
    t2 = UCS_Traversal(cost, heuristic, start_point, goals)
    t3 = A_star_Traversal(cost, heuristic, start_point, goals)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l
