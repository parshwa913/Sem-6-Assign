import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class SessionHiddenFieldServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String username = request.getParameter("username");

        // Store session data in hidden fields
        response.getWriter().println("<form action='SessionHiddenFieldServlet' method='post'>");
        response.getWriter().println("<input type='hidden' name='username' value='" + username + "'>");
        response.getWriter().println("<input type='submit' value='Continue Session'>");
        response.getWriter().println("</form>");
    }
}