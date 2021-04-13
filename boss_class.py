"""Defines a boss which stores the SLM and camera as attributes for easy
function calls which rely on both. Most of the functions in the main code are 
called from this boss object.
"""

class Boss():
    def __init__(self,slm,cam):
        self.slm = slm
        self.cam = cam

    def test(self):
        return 7