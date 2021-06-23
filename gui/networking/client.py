"""PyDex - communication over the network
Stefan Spence 29/05/20

 - Client that can send and receive data
 - note that the server should be kept running separately
"""
import socket
import struct
try:
    from PyQt4.QtCore import QThread, pyqtSignal
    from PyQt4.QtGui import QApplication
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal
    from PyQt5.QtWidgets import QApplication 
import sys
if '..' not in sys.path: sys.path.append('..')
from .mythread import reset_slot
from .strtypes import error, warning, info

def simple_msg(host, port, msg, encoding='utf-8', recv_buff_size=-1):
    """Open a socket and send a TCP message, then receive back a message."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # sock.setblocking(0) # don't hold up if there's a delay
        sock.connect((host, port))
        sock.sendall(bytes(msg, encoding))
        if recv_buff_size>0:
            return str(sock.recv(recv_buff_size), encoding)
        else: return ''
    
class PyClient(QThread):
    """Create a client that opens a socket, sends and receives data.
    Generally, you should open and close a socket for a single message.
    Running the thread will continuously try and receive a message. To stop
    the thread, set PyClient.stop = True.
    host - a string giving domain hostname or IPv4 address. 'localhost' is this.
    port - the unique port number used for the next socket connection."""
    textin = pyqtSignal(str) # received message
    dxnum = pyqtSignal(str) # received run number, synchronised with DExTer
    stop  = False           # toggle whether to stop listening
    
    def __init__(self, host='localhost', port=8089, name=''):
        super().__init__()
        self._name = name
        self.server_address = (host, port)
        self.__mq = [] # message queue
        self.app = QApplication.instance()
        self.finished.connect(self.reset_stop) # allow it to start again next time
        
    def add_message(self, enum, text, encoding="mbcs"):
        """Append a message to the queue that will be sent by TCP connection.
        enum - (int) corresponding to the enum for DExTer's producer-
                consumer loop.
        text - (str) the message to send.
        enum and message length are sent as unsigned long int (4 bytes)."""
        self.__mq.append([struct.pack("!L", int(enum)), # enum 
                                struct.pack("!L", len(bytes(text, encoding))), # msg length 
                                bytes(text, encoding)]) # message
                            
    def priority_messages(self, message_list, encoding="mbcs"):
        """Add messages to the start of the message queue.
        message_list - list of [enum (int), text(str)] pairs."""
        self.__mq = [[struct.pack("!L", int(enum)), # enum 
                            struct.pack("!L", len(bytes(text, encoding))), # msg length 
                            bytes(text, encoding)] for enum, text in message_list] + self.__mq
    
    def get_queue(self):
        """Return a list of the queued messages."""
        return [(str(int.from_bytes(enum, 'big')), int.from_bytes(tlen, 'big'), 
                str(text, 'mbcs')) for enum, tlen, text in self.__mq]
                        
    def clear_queue(self):
        """Remove all of the messages from the queue."""
        reset_slot(self.textin, self.clear_queue, False) # only trigger clear_queue once
        self.__mq = []
    
    def echo(self, encoding='mbcs'):
        """Receive and echo back 3 messages:
        1) the run number (unsigned long int)
        2) the length of a message string (unsigned long int)
        3) a message string"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect(self.server_address) # connect to server
                # sock.setblocking(1) # don't continue until msg is transferred
                # receive message
                dxn = sock.recv(4) # 4 bytes
                bytesize = sock.recv(4)# 4 bytes
                size = int.from_bytes(bytesize, 'big')
                msg = sock.recv(size)
                # send back
                if len(self.__mq):
                    try:
                        dxn, bytesize, msg = self.__mq.pop(0)
                    except IndexError as e: 
                        error('Server %s msg queue was emptied before msg could be sent.\n'%self._name+str(e))
                sock.sendall(dxn)
                sock.sendall(bytesize)
                sock.sendall(msg)
                self.dxnum.emit(str(int.from_bytes(dxn, 'big')))
                self.textin.emit(str(msg, encoding))
            except (ConnectionRefusedError, TimeoutError) as e:
                pass
            except (ConnectionResetError, ConnectionAbortedError) as e:
                error('Python client %s: server cancelled connection.\n'%self._name+str(e))
                
    def check_stop(self):
        """Check if the thread has been told to stop"""
        return self.stop
        
    def reset_stop(self):
        """Reset the stop toggle so the thread can run again"""
        self.stop = False
        
    def run(self):
        """Continuously echo back messages."""
        while not self.check_stop():
            self.app.processEvents() # make sure it doesn't block GUI events
            self.echo() # TCP msg

    def close(self, args=None):
        """Stop the event loop safely, ensuring that the sockets are closed.
        Once the thread has stopped, reset the stop toggle so that it 
        doesn't block the thread starting again the next time."""
        reset_slot(self.finished, self.reset_stop, True)
        self.stop = True