import pkgutil
import importlib
import inspect
import plugins

from plugins.base_network import BaseNetwork
def onerror(name):
    print("Error importing module %s" % name)

networks = []
# https://kagasu.hatenablog.com/entry/2021/05/03/185017
# https://stackoverflow.com/questions/17024605/python-pkgutil-walk-packages-not-returning-subpackages
for info in pkgutil.walk_packages(plugins.__path__, plugins.__name__ + '.',onerror=onerror):
    mod = importlib.import_module(info.name)
    # https://stackoverflow.com/questions/1796180/how-can-i-get-a-list-of-all-classes-within-current-module-in-python
    for name,obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and obj.__module__ == info.name:
            pass
        else:
            continue
        # https://note.nkmk.me/python-issubclass-mro-bases-subclasses/
        if BaseNetwork == obj or not issubclass(obj,BaseNetwork):
            continue
        networks += [obj]

for net in networks:
    a = net()
    print(f"Name:{a.get_name()}, type: {a.get_type()}", net)
