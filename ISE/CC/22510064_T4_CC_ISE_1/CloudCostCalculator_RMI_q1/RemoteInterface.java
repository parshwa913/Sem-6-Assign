import java.rmi.Remote;
import java.rmi.RemoteException;

public interface RemoteInterface extends Remote {
    double calculateCloudCost(double storageGB, double cpuCores, double bandwidthTB) throws RemoteException;
}