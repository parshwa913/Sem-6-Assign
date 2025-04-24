import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.Scanner;

public class RMIClient {
    public static void main(String[] args) {
        try {
            Registry registry = LocateRegistry.getRegistry("localhost", 1099);
            RemoteInterface stub = (RemoteInterface) registry.lookup("CloudCostService");

            Scanner scanner = new Scanner(System.in);

            System.out.println("Welcome Parshwa (Roll No: 22510064) to Cloud Cost Calculator");
            System.out.print("Enter Storage (GB): ");
            double storage = scanner.nextDouble();

            System.out.print("Enter CPU Cores: ");
            double cpu = scanner.nextDouble();

            System.out.print("Enter Bandwidth (TB): ");
            double bandwidth = scanner.nextDouble();

            double cost = stub.calculateCloudCost(storage, cpu, bandwidth);
            System.out.println("Estimated Cloud Cost: $" + cost);
            
            scanner.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
