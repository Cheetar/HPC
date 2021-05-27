f = open("loc-gowalla_edges.txt", "r")

edges = {}

for x in f:
    a, b = x.split("\t")
    a = int(a)
    if len(b) > 1:
        b = int(b[0])
    else:
        b = int(b)
    #print(a, b)
    if b not in edges:
        if a not in edges:
            edges[a] = []
        edges[a].append(b)
    elif a not in edges[b]:
        if a not in edges:
            edges[a] = []
        edges[a].append(b)

g = open("loc-gowalla_edges-repaired.txt", "w")
for v in range(196591):
    for adj in edges[v]:
        g.write(f"{v}\t{adj}\n")