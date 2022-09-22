import numpy as np
import torch

# a = [(1,2,3),(1,2,4),(2,3,4),(2,2,3),(3,4,5),(4,5,6),(3,4,5)]
# triple_dict = {}
# t, r =[], []
# for i,triple in enumerate(a):
#     temp_data = triple_dict.get(triple[0],[[] for j in range(2)])
#     temp_data[0].append(triple[1])
#     temp_data[1].append(triple[2])
#     triple_dict[triple[0]] = temp_data
# print(triple_dict[8])



# i_id, i_ids, pad_index, i_map, new_map = torch.load("/Users/mengyan/Documents/code/KTUP/log/myTensor.pt")
# padded_e_ids = []
# if torch.is_tensor(i_ids):
#     i_ids = i_ids.numpy()
# print(i_ids)
# for i_id in i_ids:
#     new_index = i_map[i_id]
#     ent_id = new_map[new_index][0]
#     padded_e_ids.append(ent_id if ent_id != -1 else pad_index)
# print(padded_e_ids)

# keys = [5,4,3,2,1]
# c = 12
# d = 5
# b = d*[c]
# print(b)

# a = torch.randn(5)
# b = torch.randn(5)
# c = torch.randn(5)
# print(a)
# print(a.type())
# for i in a:
#     print(i.numpy())
# c=[]
# if torch.is_tensor(c):
#     print(a.numpy())

ee = [1.22,2.33]
ee = torch.FloatTensor(ee)
y = np.array(ee).reshape(-1,len(ee))
print(y)
# d = a*ee
# print(d)
# p = torch.exp(-0.7*(1-d))
# p = 1/(1+p)
# p = torch.sum(p)

# dict = {}
# print(p)
# dict[1] =float(p)/5
# print(dict)

