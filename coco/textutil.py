def dumplist(list_data, filepath):
    with open(filepath,'w') as f:
        f.writelines(map(lambda x: str(x),list_data))