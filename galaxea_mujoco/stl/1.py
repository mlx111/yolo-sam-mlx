import trimesh

m1 = trimesh.load('pear.stl')
m2 = trimesh.load('pear1.stl')

print("pear.stl bounds:", m1.bounds)
print("pear1.stl bounds:", m2.bounds)