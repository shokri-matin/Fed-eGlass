import sys
import socket
import select
import pickle
import time

import numpy as np

from src.model import Model
from src.data_handler import DataHandler

class Server(object):

    def __init__(self, IP=socket.gethostname(), PORT=12345,
     HEADER_LENGTH=10, number_of_parties=2) -> None:
        
        self.HEADER_LENGTH = HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server_socket.bind((IP, PORT))
        self.server_socket.listen()

        self.number_of_parties = number_of_parties

        self.socket_list = []
        self.parties = {}

        print("Server initialization is completed on IP: {0}, Port: {1}".format(IP, PORT))

    def receive_message(self, _client_socket):
        try:
            message_header = _client_socket.recv(self.HEADER_LENGTH)
            if not len(message_header):
                return False
            message_length = int(message_header.decode('utf-8').strip())
            return {"header": message_header, "data": _client_socket.recv(message_length)}

        except Exception as e:
            print(f"Server error : {e}")
            return False

    def send_message(self, party, message):

        serialized_data = pickle.dumps(message)

        message_header = f"{len(serialized_data):<{self.HEADER_LENGTH}}".encode('utf-8')

        party.sendall(message_header + serialized_data)

    def accept_clients(self):

        print("Server is waiting for parties, number of parties are: {0}".format(self.number_of_parties))
        # Register all parties
        i = 1
        while i <= self.number_of_parties:
            client_socket, client_address = self.server_socket.accept()
            party = self.receive_message(client_socket)
            self.socket_list.append(client_socket)
            self.parties[client_socket] = party
            i = i + 1

            print(f"Accepted new connection from {client_address[0]} : {client_address[1]} "
                f"username: {party['data'].decode('utf-8')}")

        print("All parties are accepted!")

    def send_weights(self, weights):
        
        for party in self.parties:
            self.send_message(party, weights)

    def test(self, mediator):
        x_ts, y_ts = DataHandler().load()
        _, test_acc = mediator.model_head.evaluate(x_ts,  y_ts)

        print('\nTest accuracy:', test_acc)

class AVGServer(Server):

    def __init__(self, IP=socket.gethostname(), PORT=12345,
     HEADER_LENGTH=10, number_of_parties=2, num_of_epochs=100):

        self.num_of_epochs = num_of_epochs
        super().__init__(IP, PORT, HEADER_LENGTH, number_of_parties)

    def recv_weights(self):

        i = 0
        weights = {}
        flag = True
        while flag:
            read_sockets, _, _ = select.select(self.socket_list, [], self.socket_list)

            for notified_socket in read_sockets:
                if i != len(self.parties):
                    i = i + 1
                    message_header = notified_socket.recv(self.HEADER_LENGTH)
                    if not len(message_header):
                        return False
                    message_length = int(message_header.decode('utf-8').strip())

                    received_data = b""
                    current_length = 0
                    while current_length < message_length:
                        received_data += notified_socket.recv(message_length - current_length)
                        current_length = len(received_data)

                    weight = pickle.loads(received_data)
                    weights[notified_socket] = weight

                if i == len(self.parties):
                    flag = False

        return weights

    def aggregate(self, weights):
        parties = list(weights.keys())

        global_model_parameters = weights[parties[0]]
        for i in range(1, self.number_of_parties):
            temp = weights[parties[i]]
            for j in range(0, len(global_model_parameters)):
                global_model_parameters[j] += temp[j]

        for j in range(0, len(global_model_parameters)):
            global_model_parameters[j] /= self.number_of_parties

        return global_model_parameters

    def run(self):

        times = []

        print("create a mediator")
        # create a mediator
        mediator = Model()
        # server accepts clients
        print("server accepts clients")
        self.accept_clients()

        epoch=0
        while True:
            try:
                # get the start time
                start_time = time.time()

                # server sends its weights to clients
                weights = mediator.model_head.get_weights()
                self.send_weights(weights)

                # server waits for clients to send their weights
                cweights = self.recv_weights()

                # server aggregates weights recved from clients
                agg_cweights = self.aggregate(cweights)

                # set agg weights on mediator
                mediator.model_head.set_weights(agg_cweights)

                print('epochs:{}'.format(epoch))
                
                if epoch == self.num_of_epochs:
                    return mediator
                epoch = epoch+1

                end_time = time.time()

                times.append(end_time-start_time)

            except Exception as err:
                print(err)
                print('Server is not longer available')
                exit()

        print("mean time process : {}".format(np.mean(times)))


        return mediator

if __name__ == "__main__":
    args = sys.argv
    number_of_parties = int(args[1])
    num_of_epochs = int(args[2])

    server = AVGServer(IP=socket.gethostname(),
                         PORT=12345,
                         HEADER_LENGTH=10,
                         number_of_parties=number_of_parties, 
                         num_of_epochs=num_of_epochs)

    mediator = server.run()
    server.test(mediator)