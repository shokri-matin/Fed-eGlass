import sys
import socket
import pickle
import SMC_functions

from src.model import Model
from src.data_handler import DataHandler


class Client:

    def __init__(self, IP, PORT, HEADER_LENGTH, number_of_parties, username, batch_size, scenario) -> None:

        self.HEADER_LENGTH = HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT
        self.username = str(username)
        self.BUFFERSIZE = 1024
        self.batchsize = batch_size

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((IP, PORT))
        self.client_socket.setblocking(True)

        self.datahandler = DataHandler(username,  batch_size, number_of_parties, scenario)

        self.SMC_tools = SMC_functions.SMCtools(num_parties=number_of_parties, party_id=username-1,
                                                num_participating_parties=number_of_parties,
                                                secure_aggregation_parameter_k=number_of_parties - 1,
                                                scenario=scenario)

        print("Client initialization is completed on IP: {0}, Port: {1}".format(IP, PORT))

    def initialize_party(self):
        return Model()

    def subscribe_server(self, username):
        username = username.encode('utf-8')
        username_header = f"{len(username):<{self.HEADER_LENGTH}}".encode('utf-8')
        self.client_socket.send(username_header + username)

    def recv_weights(self):
        message_header = self.client_socket.recv(self.HEADER_LENGTH)
        if not len(message_header):
            return False

        message_length = int(message_header.decode('utf-8').strip())

        received_data = b""
        current_length = 0

        while current_length < message_length:
            received_data += self.client_socket.recv(message_length - current_length)
            current_length = len(received_data)

        weights = pickle.loads(received_data)
        return weights

class AVGClient(Client):

    def __init__(self, IP=socket.gethostname(), PORT=12345,
                    HEADER_LENGTH=10, number_of_parties=2, username=1, batch_size=16, epochs=10, scenario=1):
        
        self.epochs = epochs
        super().__init__(IP, PORT, HEADER_LENGTH, number_of_parties, username, batch_size, scenario)

    def send_weights(self, weights):

        serialized_data = pickle.dumps(weights)

        message_header = f"{len(serialized_data):<{self.HEADER_LENGTH}}".encode('utf-8')

        self.client_socket.sendall(message_header + serialized_data)

    def run(self):

        # create party model
        party = self.initialize_party()

        # subscribe server
        self.subscribe_server(self.username)

        # client waits for server request
        while True:
            try:
                # client gets mediator weights
                weights = self.recv_weights()

                # client assigns weights to model
                party.set_weights(weights)

                # updating loop
                x_train, y_train = self.datahandler.get_all_data()
                party.model_head.fit(x_train, y_train, batch_size=16, epochs=2, verbose=1)

                # get weights from party
                cweights = party.model_head.get_weights()

                masked_model_parameters = self.SMC_tools.mask(cweights)

                # client sends gradient
                self.send_weights(masked_model_parameters)

            except Exception as err:
                print(err)
                print('Server is not longer available')
                exit()

if __name__ == "__main__":
    args = sys.argv

    username = int(args[1])
    number_of_parties = int(args[2])
    batch_size = int(args[3])
    num_local_updates = int(args[4])
    scenario = int(args[5])

    client = AVGClient(number_of_parties=number_of_parties
                        , username=username, batch_size=batch_size, epochs=num_local_updates, scenario=scenario)
    client.run()
