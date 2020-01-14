from NetworkTraining import NetworkTraining

nt = NetworkTraining("Models/current_network.h5")
nt.train_network()
nt.save_after_training()
nt.save_history()

