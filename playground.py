import pickle
import time

t1 = time.time()
with open('./followers_dict_trimmed.data', 'rb') as filehandle:
    out = pickle.load(filehandle)

k = list(out.keys())
print(len(out[k[9]]))