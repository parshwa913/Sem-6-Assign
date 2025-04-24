import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class SessionCookieServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String username = request.getParameter("username");

        // Create a cookie
        Cookie userCookie = new Cookie("username", username);
        userCookie.setMaxAge(60 * 5); // 5 minutes
        response.addCookie(userCookie);

        response.getWriter().println("Session stored in Cookie: " + username);
    }
}
