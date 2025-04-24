 
import java.rmi.Naming;

public class RMIClient {
    public static void main(String[] args) {
        try {
            RemoteInterface stub = (RemoteInterface) Naming.lookup("rmi://localhost/HelloService");
            String response = stub.sayHello("User");
            System.out.println("Response from Server: " + response);
        } catch (Exception e) {
            System.err.println("Client Exception: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
