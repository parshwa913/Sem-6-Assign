 
import java.rmi.Naming;
import java.rmi.registry.LocateRegistry;

public class RMIServer {
    public static void main(String[] args) {
        try {
            LocateRegistry.createRegistry(1099); // Start RMI Registry
            RemoteImplementation obj = new RemoteImplementation();
            Naming.rebind("rmi://localhost/HelloService", obj);
            System.out.println("RMI Server is running...");
        } catch (Exception e) {
            System.err.println("Server Exception: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
