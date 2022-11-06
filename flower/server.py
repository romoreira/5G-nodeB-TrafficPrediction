import flwr as fl

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=1))
print(type(fl.server.Server.evaluate_round(1)))