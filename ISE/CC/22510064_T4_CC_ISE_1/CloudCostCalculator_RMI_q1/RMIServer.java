import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class RMIServer {
    public static void main(String[] args) {
        try {
            RemoteImplementation obj = new RemoteImplementation();
            
            Registry registry = LocateRegistry.createRegistry(1099);
            
            registry.rebind("CloudCostService", obj);
            
            System.out.println("RMI Server is running...");
        } catch (Exception e) {
            System.err.println("Server Exception: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
