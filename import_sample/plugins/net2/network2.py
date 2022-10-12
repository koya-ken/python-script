from ..base_network import BaseNetwork

class Net2(BaseNetwork):
    def get_name(self) -> str:
        return "Net2"
    
    def get_type(self) -> str:
        return "cnn"