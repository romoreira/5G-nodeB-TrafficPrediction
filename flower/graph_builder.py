import matplotlib.pyplot as plt
import os

resultados_dir = './Resultados/' + str("OmniScaleCNN")
if os.path.isdir(resultados_dir) == False:
    os.mkdir(resultados_dir)

def create_loss_graph(train_losses, test_losses, plt_title):
    # Cria os graficos de decaimento treino e validação (imprime na tela e salva na pasta "./Resultados")
    plt.title("Training Loss")
    plt.plot(train_losses, label='Train')
    #plt.plot(test_losses, label='Test')
    plt.legend(frameon=False)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(b=None)
    plt.legend()
    plt.grid()
    plt.savefig(resultados_dir + '/' + 'graf_' + str(plt_title) + '.pdf')
    plt.close()


list = [0.0006345268338918686, 0.0004741679294966161, 0.0004362062318250537, 0.000420869211666286, 0.0004127761349081993, 0.000398952019168064, 0.0003970783727709204, 0.000392532761907205, 0.00038986571598798037, 0.00038960183155722916, 0.0003934527048841119, 0.00039658730383962393, 0.00038851459976285696, 0.00038265620241872966, 0.0003804049047175795, 0.00037885940400883555, 0.0003771291521843523, 0.0003688946017064154, 0.00038107787258923054, 0.0003784937725868076, 0.0003766869776882231, 0.0003770127659663558, 0.000367439235560596, 0.0003720287641044706, 0.0003654650063253939, 0.00037455270648933947, 0.00037904366035945714, 0.00036829852615483105, 0.0003860894066747278, 0.00037411769153550267]

create_loss_graph(list, [], str("Loss Train" + str(2)))
