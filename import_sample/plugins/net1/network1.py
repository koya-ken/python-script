from ..base_network import BaseNetwork

class Net1(BaseNetwork):
    def get_name(self) -> str:
        return "Net1"
    
    def get_type(self) -> str:
        return "cnn"