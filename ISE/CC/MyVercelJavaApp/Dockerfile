# File: Dockerfile
FROM openjdk:11-jre-slim

# Set working directory inside the container
WORKDIR /app

# Copy the Java source file from the "api" folder into the container
COPY api/cost.java .

# Compile the Java source file
RUN javac cost.java

# Expose the port (Vercel sets the PORT environment variable; default is 3000)
EXPOSE 3000

# Run the Java application
CMD ["java", "cost"]
