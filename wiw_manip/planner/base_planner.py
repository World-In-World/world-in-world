class BasePlanner:
    def __init__(self, **kwargs):
        if kwargs:
            print("***Warning: Planner base class received unused arguments:", kwargs)
        pass
    
    def reset(self, **kwargs):
        pass
    
    def update_info(self, info, **kwargs):
        pass