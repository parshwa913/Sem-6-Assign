import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

public class RemoteImplementation extends UnicastRemoteObject implements RemoteInterface {
    
    protected RemoteImplementation() throws RemoteException {
        super();
    }
    
    @Override
    public double calculateCloudCost(double storageGB, double cpuCores, double bandwidthTB) throws RemoteException {
        double storageCost = storageGB * 0.02;    
        double cpuCost = cpuCores * 5.0;           
        double bandwidthCost = bandwidthTB * 10.0;  
        return storageCost + cpuCost + bandwidthCost;
    }
}
