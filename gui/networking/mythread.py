"""PyDex - Experimental control using Python and DExTer
Stefan Spence 06/12/19

 - A generic thread.
 - the thread is started by instantiating it and calling start().
 - implement a queue of items to act on.
 - when the thread is running, it iterates through items in the queue.
 - if the queue is empty, keep refreshing.
 - stop the thread by calling close()
"""
try:
    from PyQt4.QtCore import QThread, pyqtSignal
    from PyQt4.QtGui import QApplication
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal
    from PyQt5.QtWidgets import QApplication 

def reset_slot(signal, slot, reconnect=True):
    """Make sure all instances of slot are disconnected
    from signal. Prevents multiple connections to the same 
    slot. If reconnect=True, then reconnect slot to signal."""
    while True: # make sure that the slot is only connected once 
        try: signal.disconnect(slot)
        except TypeError: break
    if reconnect: signal.connect(slot)

class PyDexThread(QThread):
    """A template thread that continuously iterates an action 
    on a FIFO queue of items."""
    stop  = False # toggle to stop the thread running
    queue = []    # list of items to process

    def __init__(self):
        super().__init__()
        self.app = QApplication.instance()

    def add_item(self, new_item, *args, **kwargs):
        """Append a new item to the queue for processing."""
        self.queue.append(new_item)

    def process(self, item, *args, **kwargs):
        """Process an item in the queue."""
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """Run the thread continuously processing items
        from the queue until the stop bool is toggled."""
        while True:
            self.app.processEvents() # avoids GUI lag but can cause events to be missed
            if self.check_stop():
                break # stop the thread running
            elif len(self.queue):
                self.process(self.queue.pop(0), *args, **kwargs)

    def check_stop(self):
        """Check the value of stop - must be a function in order to work in
        a while loop."""
        return self.stop
        
    def reset_stop(self):
        """Reset the stop toggle so that the event loop can run."""
        self.stop = False
    
    def close(self):
        """Stop the thread safely. Once the thread has stopped, 
        reset the stop toggle so that it doesn't block the 
        thread from starting again the next time."""
        reset_slot(self.finished, self.reset_stop)
        self.stop = True