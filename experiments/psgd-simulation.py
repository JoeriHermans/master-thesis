"""Parallel SGD simulation for development purposes.

Author: Joeri R. Hermans
"""

## BEGIN Imports. ##############################################################

import argparse

import cPickle as pickle

import copy

import math

import matplotlib as mlab
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np

import socket

import sys

import os

import signal

import time

import threading

## END Imports. ################################################################

## BEGIN Globals. ##############################################################

# Total number of workers.
g_num_workers = 10

# Parameter server port.
g_port_ps = 5000

# Worker list.
g_workers = []

# Parameter server.
g_parameter_server = None

# Parameter space
r = 100
X = np.linspace(-r, r, 1000)
X, Y = np.meshgrid(X, X)
Z = (X + 2*Y - 7)**2 + (2 * X + Y - 5)**2

## END Globals. ################################################################

def get_position(model):
    return (model[0] + 2 * model[1] - 7)**2 + (2 * model[0] + model[1] - 5)**2

def get_gradient(model):
    x = model[0]
    y = model[1]
    gradient_x = 10 * x + 8 * y - 34
    gradient_y = 8 * x + 10 * y - 38

    return np.asarray([gradient_x, gradient_y])

def allocate_model():
    model = np.asarray([90.0, 90.0])

    return model

def signal_sigint(signal, frame):
    print("SIGINT received")
    os.system('kill %d' % os.getpid())

def catch_signals():
    signal.signal(signal.SIGINT, signal_sigint)

def allocate_settings():
    settings = {}
    settings['communication_window'] = 5
    settings['learning_rate'] = 0.0001

    return settings

def process_arguments():
    global g_num_workers
    global g_port_ps

    for i in range(0, len(sys.argv)):
        arg = sys.argv[i]
        if arg == '--num-workers':
            g_num_workers = int(sys.argv[i + 1])
        elif arg == '--port':
            g_port_ps = int(sys.argv[i + 1])

def allocate_parameter_server(model):
    global g_parameter_server
    global g_port_ps

    g_parameter_server = parameter_server(model, g_port_ps)
    g_parameter_server.start()

def allocate_workers(model):
    global g_workers
    global g_num_workers
    global g_port_ps

    # Allocate the workers.
    for i in range(0, g_num_workers):
        settings = allocate_settings()
        w = worker(model, g_port_ps, settings)
        g_workers.append(w)
    # Start the workers.
    for w in g_workers:
        w.start()

def update_visualization():
    global g_num_workers
    global g_parameter_server
    global g_workers
    global X
    global Y
    global Z
    global r

    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z, zorder=0)
    data = np.zeros((g_num_workers + 1, 2))
    for i in range(0, g_num_workers):
        w = g_workers[i]
        m = w.get_model()
        data[i] = m
    data[g_num_workers] = g_parameter_server.get_center_variable()
    scatter = ax.scatter(data[:, 0], data[:, 1], zorder=1)

    def update_visualization_raw(iteration):
        # Update the data array.
        for i in range(0, g_num_workers):
            w = g_workers[i]
            m = w.get_model()
            data[i] = m
        data[g_num_workers] = g_parameter_server.get_center_variable()
        colors = np.full(g_num_workers + 1, "b", dtype=np.dtype(str))
        colors[g_num_workers] = "r"
        scatter.set_color(colors)
        scatter.set_offsets(data)

        return scatter,

    anim = animation.FuncAnimation(fig, update_visualization_raw, interval=5, blit=True)
    plt.xlim([-r, r])
    plt.ylim([-r, r])
    plt.show()

def main():
    global g_num_workers
    global g_port_ps

    catch_signals()
    process_arguments()
    model = allocate_model()
    allocate_parameter_server(model)
    allocate_workers(model)
    update_visualization()
    signal.pause()

## BEGIN Classes. ##############################################################


class worker(threading.Thread):

    def __init__(self, model, ps_port, settings):
        threading.Thread.__init__(self)
        self.model = model
        self.center_variable = None
        self.parameter_server_port = ps_port
        self.settings = settings
        self.mutex = threading.Lock()
        self.socket = None
        self.running = False

    def get_model(self):
        with self.mutex:
            model = copy.deepcopy(self.model)

        return model

    def connect(self):
        fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        fd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        fd.connect(("localhost", self.parameter_server_port))

        self.socket = fd

    def commit(self, delta):
        data = {}
        data['delta'] = delta
        self.socket.sendall(b'c')
        send_data(self.socket, data)

    def pull(self):
        self.socket.sendall(b'p')
        cv = np.asarray(recv_data(self.socket))

        return cv

    def start(self):
        self.running = True
        self.connect()
        super(worker, self).start()

    def process_gradients(self):
        communication_window = self.settings['communication_window']
        learning_rate = self.settings['learning_rate']
        self.center_variable = self.pull()
        update = 0
        delta = np.asarray([0.0, 0.0])
        while self.running:
            g = get_gradient(self.model)
            delta -= learning_rate * g
            # Apply gradient update.
            if update % communication_window == 0:
                cv = self.pull()
                #delta /= communication_window
                #t = np.abs(cv - self.center_variable)
                #t = 1 / (np.exp(t))
                #delta = np.multiply(t, delta)
                self.commit(delta)
                delta.fill(0.0)
                cv += delta
                #cv = self.pull()
                self.center_variable = cv
                self.model = cv
            # Increment update counter.
            update += 1

    def run(self):
        time.sleep(2)
        self.process_gradients()

class parameter_server(threading.Thread):

    def __init__(self, center_variable, port):
        threading.Thread.__init__(self)
        self.center_variable = center_variable
        self.port = port
        self.mutex = threading.Lock()
        self.connections = []
        self.running = False
        self.socket = None

    def get_center_variable(self):
        with self.mutex:
            cv = copy.deepcopy(self.center_variable)

        return cv

    def initialize(self):
        self.running = True
        fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        fd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        fd.bind(('0.0.0.0', self.port))
        fd.listen(5)
        self.socket = fd

    def handle_commit(self, conn, addr):
        data = recv_data(conn)
        delta = data['delta']
        with self.mutex:
            self.center_variable += delta
            print("Error: " + str(get_position(self.center_variable)))

    def handle_pull(self, conn, addr):
        with self.mutex:
            cv = self.center_variable
            cv = copy.deepcopy(cv)
        send_data(conn, cv)

    def cancel_accpt(self):
        fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            fd.connect(("localhost", self.port))
            fd.close()
        except Exception:
            pass

    def handle_connection(self, conn, addr):
        while self.running:
            action = conn.recv(1).decode()
            if action == 'c':
                self.handle_commit(conn, addr)
            else:
                self.handle_pull(conn, addr)

    def start(self):
        self.stop()
        self.initialize()
        super(parameter_server, self).start()

    def run(self):
        while self.running:
            try:
                conn, addr = self.socket.accept()
                thread = threading.Thread(target=self.handle_connection, args=(conn,addr))
                thread.start()
                self.connections.append(thread)
            except:
                pass

    def stop(self):
        self.running = False
        if self.socket:
            self.cleanup_connections()
            self.socket.close()
            self.cancel_accept()
            self.socket = None
        self.connections = []

    def cleanup_connections(self):
        for thread in self.connections:
            thread.join()
            del thread


## END Classes. ################################################################

## BEGIN Utility methods. ######################################################

def recvall(connection, num_bytes):
    """Reads `num_bytes` bytes from the specified connection.

    # Arguments
        connection: socket. Opened socket.
        num_bytes: int. Number of bytes to read.
    """
    byte_buffer = ''
    buffer_size = 0
    bytes_left = num_bytes
    # Iterate until we received all data.
    while buffer_size < num_bytes:
        # Fetch the next frame from the network.
        data = connection.recv(bytes_left)
        # Compute the size of the frame.
        delta = len(data)
        buffer_size += delta
        bytes_left -= delta
        # Append the data to the buffer.
        byte_buffer += data

    return byte_buffer


def recv_data(connection):
    """Will fetch the next data frame from the connection.

    The protocol for reading is structured as follows:
    1. The first 20 bytes represents a string which holds the next number of bytes to read.
    2. We convert the 20 byte string to an integer (e.g. '00000000000000000011' -> 11).
    3. We read `num_bytes` from the socket (which is in our example 11).
    4. Deserialize the retrieved string.

    # Arguments
        connection: socket. Opened socket.
    """
    data = ''
    # Fetch the serialized data length.
    length = int(recvall(connection, 20).decode())
    # Fetch the serialized data.
    serialized_data = recvall(connection, length)
    # Deserialize the data.
    data = pickle.loads(serialized_data)

    return data


def send_data(connection, data):
    """Sends the data to the other endpoint of the socket using our protocol.

    The protocol for sending is structured as follows:
    1. Serialize the data.
    2. Obtain the buffer-size of the serialized data.
    3. Serialize the buffer-size in 20 bytes (e.g. 11 -> '00000000000000000011').
    4. Send the serialized buffer size.
    5. Send the serialized data.

    # Arguments
        connection: socket. Opened socket.
        data: any. Data to send.
    """
    # Serialize the data.
    serialized_data = pickle.dumps(data, -1)
    length = len(serialized_data)
    # Serialize the number of bytes in the data.
    serialized_length = str(length).zfill(20)
    # Send the data over the provided socket.
    connection.sendall(serialized_length.encode())
    connection.sendall(serialized_data)


def connect(host, port, disable_nagle=True):
    fd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Check if Nagle's algorithm needs to be disabled.
    if disable_nagle:
        fd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    else:
        fd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)
    # Connect to the specified URI.
    fd.connect((host, port))

    return fd


def determine_host_address():
    """Determines the human-readable host address of the local machine."""
    host_address = socket.gethostbyname(socket.gethostname())

    return host_address

## END Utility methods. ########################################################

if __name__ == '__main__':
    main()
