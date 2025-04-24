// File: api/cost.java
import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;

public class cost {
    public static void main(String[] args) throws Exception {
        // Get the port from the environment variable; default to 3000 if not set
        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "3000"));
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/api/cost", new CostHandler());
        server.setExecutor(null); // uses a default executor
        server.start();
        System.out.println("Java server is listening on port " + port);
    }

    static class CostHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            // Parse query parameters (e.g., ?storage=100&cpu=4&bandwidth=2)
            String query = exchange.getRequestURI().getQuery();
            double storage = 0, cpu = 0, bandwidth = 0;
            if (query != null) {
                String[] params = query.split("&");
                for (String param : params) {
                    String[] pair = param.split("=");
                    if (pair.length == 2) {
                        switch (pair[0]) {
                            case "storage":
                                storage = Double.parseDouble(pair[1]);
                                break;
                            case "cpu":
                                cpu = Double.parseDouble(pair[1]);
                                break;
                            case "bandwidth":
                                bandwidth = Double.parseDouble(pair[1]);
                                break;
                        }
                    }
                }
            }
            // Calculate cost:
            // Storage: $0.02 per GB, CPU: $5 per core, Bandwidth: $10 per TB.
            double costCalc = (storage * 0.02) + (cpu * 5) + (bandwidth * 10);
            String jsonResponse = "{\"cost\":\"" + String.format("%.2f", costCalc) + "\"}";
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, jsonResponse.getBytes().length);
            OutputStream os = exchange.getResponseBody();
            os.write(jsonResponse.getBytes());
            os.close();
        }
    }
}
